from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .models import (
    BoundaryLevel,
    CompatibilityExpectation,
    InterfaceChangeException,
    InterfaceChangeType,
    ProductSpec,
    RaisedBy,
    ReuseSearchReport,
    Urgency,
)
from .registry import CodeIndex
from .settings import RuntimeSettings
from .state_store import project_scoped_root
from .utils import emit_structured_spec, validate_spec

logger = logging.getLogger(__name__)

_BRIGHTDATA_ENDPOINT = "https://api.brightdata.com/request"
_BRIGHTDATA_TIMEOUT_SECONDS = 30
_SERP_MAX_RESULTS = 5
_RESULT_KEYS = ("organic", "organic_results", "results")
_RESULT_FIELD_SIGNALS = ("url", "link", "title", "description", "snippet")


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    """Send a JSON POST request and return the parsed JSON response.

    Args:
        url: The endpoint URL to POST to.
        payload: JSON-serializable dictionary to send as the request body.
        headers: Additional HTTP headers (merged with Content-Type).

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        RuntimeError: If the HTTP request fails or the response is not valid JSON.
    """
    request = urllib.request.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(request, timeout=_BRIGHTDATA_TIMEOUT_SECONDS) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.error(
            "HTTP %d from %s: %s",
            exc.code,
            url,
            body,
        )
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        logger.error("URL error reaching %s: %s", url, exc.reason)
        raise RuntimeError(
            f"Failed to reach {url}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON response from %s", url)
        raise RuntimeError(
            f"Invalid JSON response from {url}"
        ) from exc


def _project_root_from_env() -> Path:
    """Derive the project-scoped state store root from environment settings.

    Returns:
        Absolute path to the project-scoped state store root directory.
    """
    settings = RuntimeSettings.from_env()
    return project_scoped_root(Path(settings.state_store_root), settings.project_id)


def _coerce_json_object(payload: Any) -> dict[str, Any]:
    """Normalize a Bright Data response into a dictionary.

    Handles three cases: direct dict, list (wrapped in ``{"results": ...}``),
    or JSON string that parses to one of those types.

    Args:
        payload: Raw response from the Bright Data SERP API.

    Returns:
        Normalized dictionary representation.

    Raises:
        RuntimeError: If the payload cannot be coerced to a JSON object.
    """
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"results": payload}
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Bright Data response was not valid JSON") from exc
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"results": parsed}
    raise RuntimeError("Bright Data response must be a JSON object or JSON list")


def _first_string(entry: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty string value found among ``keys`` in ``entry``.

    Args:
        entry: A dictionary to search.
        keys: Ordered tuple of keys to check.

    Returns:
        The first non-empty stripped string value, or empty string if none found.
    """
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _find_result_candidates(payload: Any) -> list[dict[str, Any]]:
    """BFS through a nested payload to find the best list of SERP result dicts.

    Looks for keys ``organic``, ``organic_results``, or ``results`` containing
    lists of dictionaries. Falls back to heuristic detection of dicts containing
    common SERP fields (url, link, title, description, snippet).

    Args:
        payload: Nested JSON structure from the Bright Data SERP API.

    Returns:
        List of result dictionaries, or empty list if none found.
    """
    queue: deque[Any] = deque([payload])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            for key in _RESULT_KEYS:
                value = node.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            for value in node.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
            continue
        if isinstance(node, list):
            dict_items = [item for item in node if isinstance(item, dict)]
            if dict_items and any(
                any(field in item for field in _RESULT_FIELD_SIGNALS)
                for item in dict_items[:5]
            ):
                return dict_items
            queue.extend(item for item in node if isinstance(item, (dict, list)))
    return []


def _compact_bright_data_results(
    payload: dict[str, Any],
    *,
    max_results: int,
) -> list[dict[str, str]]:
    """Extract and normalize SERP results into compact url/title/snippet dicts.

    Args:
        payload: Normalized Bright Data response dictionary.
        max_results: Maximum number of results to return.

    Returns:
        List of compact result dicts with keys ``url``, ``title``, ``snippet``.
    """
    candidates = _find_result_candidates(payload)
    compact: list[dict[str, str]] = []
    for entry in candidates:
        url = _first_string(entry, ("url", "link", "displayed_link"))
        title = _first_string(entry, ("title", "name", "headline"))
        snippet = _first_string(entry, ("snippet", "description", "text", "body"))
        if not any((url, title, snippet)):
            continue
        compact.append(
            {
                "url": url,
                "title": title,
                "snippet": snippet,
            }
        )
        if len(compact) >= max_results:
            break
    return compact


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web via the Bright Data SERP API and return compact JSON results.

    Sends a real HTTP POST to ``https://api.brightdata.com/request`` using the
    configured SERP zone and API key. Returns up to 5 organic results as JSON
    with ``url``, ``title``, and ``snippet`` for each result.

    Required environment variables:
        BRIGHTDATA_API_KEY: Bearer token for Bright Data API authentication.
        BRIGHTDATA_SERP_ZONE: Zone name configured in the Bright Data dashboard.

    Optional environment variables:
        BRIGHTDATA_SERP_COUNTRY: Two-letter country code (default: ``us``).

    Args:
        query: The search query string.

    Returns:
        JSON string containing ``provider``, ``query``, and ``results`` array.

    Raises:
        RuntimeError: If required environment variables are missing or the API
            request fails.
    """
    brightdata_key = os.getenv("BRIGHTDATA_API_KEY", "").strip()
    if not brightdata_key:
        raise RuntimeError("web_search requires BRIGHTDATA_API_KEY")
    brightdata_zone = os.getenv("BRIGHTDATA_SERP_ZONE", "").strip()
    if not brightdata_zone:
        raise RuntimeError("web_search requires BRIGHTDATA_SERP_ZONE")
    country = os.getenv("BRIGHTDATA_SERP_COUNTRY", "us").strip() or "us"
    search_url = f"https://www.google.com/search?{urllib.parse.urlencode({'q': query})}"
    payload: dict[str, str] = {
        "zone": brightdata_zone,
        "url": search_url,
        "format": "json",
        "country": country,
        "method": "GET",
    }
    logger.debug("web_search query=%r zone=%s country=%s", query, brightdata_zone, country)
    data = _http_post_json(
        _BRIGHTDATA_ENDPOINT,
        payload,
        headers={"Authorization": f"Bearer {brightdata_key}"},
    )
    normalized = _coerce_json_object(data)
    results = _compact_bright_data_results(normalized, max_results=_SERP_MAX_RESULTS)
    logger.debug("web_search returned %d results for query=%r", len(results), query)
    return json.dumps(
        {
            "provider": "brightdata",
            "query": query,
            "results": results,
        },
        indent=2,
    )


@tool("validate_spec")
def validate_spec_tool(spec_path: str) -> str:
    """Validate a ProductSpec JSON file against schema and completeness criteria.

    Reads the JSON file at ``spec_path``, deserializes it into a ``ProductSpec``,
    and runs the full validation suite. Returns a ``ValidationReport`` as JSON
    containing any errors and warnings.

    Args:
        spec_path: Filesystem path to a ProductSpec JSON file.

    Returns:
        JSON string of the ``ValidationReport`` with ``errors`` and ``warnings``.

    Raises:
        FileNotFoundError: If ``spec_path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        pydantic.ValidationError: If the JSON does not match ``ProductSpec`` schema.
    """
    path = Path(spec_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = ProductSpec.model_validate(payload)
    report = validate_spec(spec)
    return report.model_dump_json(indent=2)


@tool("emit_structured_spec")
def emit_structured_spec_tool(spec_json: str, output_path: str) -> str:
    """Persist a validated ProductSpec JSON artifact to disk.

    Deserializes ``spec_json`` into a ``ProductSpec``, validates it, and writes
    the result to ``output_path``. Validation errors cause the emission to be
    blocked with a ``ValueError``.

    Args:
        spec_json: JSON string representing a valid ``ProductSpec``.
        output_path: Filesystem path where the spec will be written.

    Returns:
        The ``output_path`` string on success.

    Raises:
        ValueError: If the spec fails validation.
        pydantic.ValidationError: If ``spec_json`` does not match ``ProductSpec``.
    """
    spec = ProductSpec.model_validate_json(spec_json)
    path = Path(output_path)
    emit_structured_spec(spec, path)
    return str(path)


@tool("search_code_index")
def search_code_index(
    query: str,
    top_k: int = 10,
    include_inactive: bool = False,
    level: str | None = None,
    tags: list[str] | None = None,
    input_types: list[str] | None = None,
    output_types: list[str] | None = None,
) -> str:
    """Search the Chroma-backed code index and return ranked results as JSON.

    Performs a semantic similarity search against the project code index using
    the configured embedding model. Supports filtering by boundary level, tags,
    and I/O types.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return. Defaults to 10.
        include_inactive: Whether to include deprecated/superseded entries.
        level: Optional ``BoundaryLevel`` string filter (e.g. ``L4_COMPONENT``).
        tags: Optional list of tags to filter by (intersection match).
        input_types: Optional list of input type strings to filter by.
        output_types: Optional list of output type strings to filter by.

    Returns:
        JSON string containing a list of ``{entry, similarity_score}`` objects.
    """
    settings = RuntimeSettings.from_env()
    index = CodeIndex(
        _project_root_from_env() / "code_index",
        embedding_model=settings.embedding_model,
    )
    matches = index.search(
        query,
        top_k=top_k,
        include_inactive=include_inactive,
        level=BoundaryLevel(level) if level else None,
        tags=tags,
        input_types=input_types,
        output_types=output_types,
    )
    results = [
        {
            "entry": match.entry.model_dump(mode="json"),
            "similarity_score": match.similarity_score,
        }
        for match in matches
    ]
    return json.dumps(results, indent=2)


@tool("raise_interface_change_exception")
def raise_interface_change_exception(
    target_module: str,
    reason: str,
    evidence: list[str],
    proposed_delta: dict[str, Any],
    compatibility: str,
    urgency: str,
    role: str = "proposer",
    run_id: str = "local-run",
) -> str:
    """Raise and persist an Interface Change Exception (ICE).

    Creates an ``InterfaceChangeException`` artifact, assigns it a unique ID,
    validates all fields via Pydantic, and writes it to the project exceptions
    directory. Returns a JSON confirmation with ``task_halted: true``.

    Args:
        target_module: The module_ref of the module whose interface is changing.
        reason: Human-readable explanation of why the change is needed.
        evidence: List of evidence strings supporting the change request.
        proposed_delta: Dictionary describing the proposed contract changes.
            Must conform to ``ProposedContractDelta`` schema.
        compatibility: Expected compatibility level (``BACKWARD_COMPATIBLE``,
            ``BREAKING``, or ``UNKNOWN``).
        urgency: Priority level (``BLOCKING``, ``HIGH``, ``NORMAL``, or ``LOW``).
        role: Role of the agent raising the exception. Defaults to ``proposer``.
        run_id: Identifier for the current run. Defaults to ``local-run``.

    Returns:
        JSON string with ``exception_id``, ``status``, and ``task_halted`` fields.

    Raises:
        pydantic.ValidationError: If any field fails schema validation.
    """
    exception = InterfaceChangeException(
        exception_id=f"ICE-{os.urandom(4).hex()}",
        type=InterfaceChangeType.INCOMPLETE,
        raised_by=RaisedBy(
            artifact_ref=f"{role}:{datetime.now(UTC).isoformat()}",
            role=role,
            run_id=run_id,
        ),
        target_module=target_module,
        reason=reason,
        evidence=evidence,
        proposed_contract_delta=proposed_delta,
        compatibility_expectation=CompatibilityExpectation(compatibility),
        urgency=Urgency(urgency),
    )
    path = _project_root_from_env() / "exceptions" / f"{exception.exception_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(exception.model_dump_json(indent=2), encoding="utf-8")
    logger.info("ICE raised: %s for module %s", exception.exception_id, target_module)
    return json.dumps({"exception_id": exception.exception_id, "status": "RAISED", "task_halted": True})


@tool("create_reuse_search_report")
def create_reuse_search_report(
    query: str,
    candidates_considered: list[dict[str, str]],
    conclusion: str,
    justification: str,
) -> str:
    """Validate and return a ReuseSearchReport as JSON.

    Constructs a ``ReuseSearchReport`` from the provided arguments, validating
    all fields via the Pydantic model. The ``candidates_considered`` dicts are
    coerced into ``ReuseSearchCandidate`` instances by Pydantic.

    Args:
        query: The search query that was used to find reuse candidates.
        candidates_considered: List of dicts with ``module_ref`` and
            ``why_rejected`` keys describing each considered candidate.
        conclusion: Reuse conclusion (``REUSE_EXISTING`` or ``CREATE_NEW``).
        justification: Explanation for the conclusion. Required when
            ``conclusion`` is ``CREATE_NEW``.

    Returns:
        JSON string of the validated ``ReuseSearchReport``.

    Raises:
        pydantic.ValidationError: If validation fails.
    """
    report = ReuseSearchReport(
        query=query,
        candidates_considered=candidates_considered,
        conclusion=conclusion,
        justification=justification,
    )
    return report.model_dump_json(indent=2)
