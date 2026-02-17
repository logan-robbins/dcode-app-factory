from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .models import (
    CompatibilityExpectation,
    InterfaceChangeException,
    InterfaceChangeType,
    RaisedBy,
    ReuseSearchReport,
    Urgency,
)
from .registry import AppendOnlyCodeIndex
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


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web using Tavily or SerpAPI and return compact JSON results."""
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        payload = {
            "api_key": tavily_key,
            "query": query,
            "max_results": 5,
            "include_answer": True,
        }
        data = _http_post_json("https://api.tavily.com/search", payload, headers={})
        return json.dumps(data, indent=2)

    serp_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if serp_key:
        url = (
            "https://serpapi.com/search.json?"
            + urllib.parse.urlencode({"q": query, "api_key": serp_key, "engine": "google"})
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")

    raise RuntimeError("web_search requires TAVILY_API_KEY or SERPAPI_API_KEY")


@tool("validate_spec")
def validate_spec_tool(spec_path: str) -> str:
    """Validate ProductSpec JSON against schema and completeness criteria."""
    path = Path(spec_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    from .models import StructuredSpec

    spec = StructuredSpec.model_validate(payload)
    report = validate_spec(spec)
    return report.model_dump_json(indent=2)


@tool("emit_structured_spec")
def emit_structured_spec_tool(spec_json: str, output_path: str) -> str:
    """Persist validated ProductSpec JSON artifact."""
    from .models import StructuredSpec

    spec = StructuredSpec.model_validate_json(spec_json)
    path = Path(output_path)
    emit_structured_spec(spec, path)
    return str(path)


@tool("search_code_index")
def search_code_index(
    query: str,
    top_k: int = 10,
    include_inactive: bool = False,
    tags: list[str] | None = None,
    input_types: list[str] | None = None,
    output_types: list[str] | None = None,
) -> str:
    """Search append-only code index and return ranked JSON results."""
    settings = RuntimeSettings.from_env()
    index = AppendOnlyCodeIndex(
        _project_root_from_env() / "code_index",
        embedding_model=settings.embedding_model,
    )
    matches = index.search(
        query,
        top_k=top_k,
        include_inactive=include_inactive,
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
