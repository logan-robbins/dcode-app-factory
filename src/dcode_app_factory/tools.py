from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
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


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read().decode("utf-8")
        return json.loads(data)


def _project_root_from_env() -> Path:
    settings = RuntimeSettings.from_env()
    return project_scoped_root(Path(settings.state_store_root), settings.project_id)


def _coerce_json_object(payload: Any) -> dict[str, Any]:
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
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _find_result_candidates(payload: Any) -> list[dict[str, Any]]:
    queue: list[Any] = [payload]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            for key in ("organic", "organic_results", "results"):
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
                any(field in item for field in ("url", "link", "title", "description", "snippet"))
                for item in dict_items[:5]
            ):
                return dict_items
            queue.extend(item for item in node if isinstance(item, (dict, list)))
    return []


def _compact_bright_data_results(payload: dict[str, Any], *, max_results: int) -> list[dict[str, str]]:
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
    """Search the web using Bright Data and return compact JSON results."""
    brightdata_key = os.getenv("BRIGHTDATA_API_KEY", "").strip()
    if not brightdata_key:
        raise RuntimeError("web_search requires BRIGHTDATA_API_KEY")
    brightdata_zone = os.getenv("BRIGHTDATA_SERP_ZONE", "").strip()
    if not brightdata_zone:
        raise RuntimeError("web_search requires BRIGHTDATA_SERP_ZONE")
    country = os.getenv("BRIGHTDATA_SERP_COUNTRY", "us").strip() or "us"
    search_url = f"https://www.google.com/search?{urllib.parse.urlencode({'q': query})}"
    payload = {
        "zone": brightdata_zone,
        "url": search_url,
        "format": "json",
        "country": country,
        "method": "GET",
    }
    data = _http_post_json(
        "https://api.brightdata.com/request",
        payload,
        headers={"Authorization": f"Bearer {brightdata_key}"},
    )
    normalized = _coerce_json_object(data)
    results = _compact_bright_data_results(normalized, max_results=5)
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
    """Validate ProductSpec JSON against schema and completeness criteria."""
    path = Path(spec_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = ProductSpec.model_validate(payload)
    report = validate_spec(spec)
    return report.model_dump_json(indent=2)


@tool("emit_structured_spec")
def emit_structured_spec_tool(spec_json: str, output_path: str) -> str:
    """Persist validated ProductSpec JSON artifact."""
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
    """Search code index and return ranked JSON results."""
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
    payload = [
        {
            "entry": match.entry.model_dump(mode="json"),
            "similarity_score": match.similarity_score,
        }
        for match in matches
    ]
    return json.dumps(payload, indent=2)


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
    """Raise and persist an interface change exception (ICE)."""
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
    return json.dumps({"exception_id": exception.exception_id, "status": "RAISED", "task_halted": True})


@tool("create_reuse_search_report")
def create_reuse_search_report(query: str, candidates_considered: list[dict[str, str]], conclusion: str, justification: str) -> str:
    """Validate and return a ReuseSearchReport."""
    report = ReuseSearchReport(
        query=query,
        candidates_considered=candidates_considered,
        conclusion=conclusion,
        justification=justification,
    )
    return report.model_dump_json(indent=2)
