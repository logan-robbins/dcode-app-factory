from __future__ import annotations

import hashlib
import json
import urllib.error
from pathlib import Path

import pytest

from dcode_app_factory import tools as tools_module
from dcode_app_factory.models import (
    ArtifactStatus,
    CompatibilityRule,
    ContractErrorSurface,
    ContractInput,
    ContractModes,
    ContractOutput,
    EffectType,
    MicroModuleContract,
    RuntimeBudgets,
)
from dcode_app_factory.registry import CodeIndex
from dcode_app_factory.settings import RuntimeSettings
from dcode_app_factory.state_store import FactoryStateStore, project_scoped_root


@pytest.fixture(autouse=True)
def _deterministic_embeddings_for_tools_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tools tests use deterministic embeddings, matching the test_factory fixture."""
    monkeypatch.setenv("FACTORY_EMBEDDING_MODEL", "text-embedding-3-large")

    class _DeterministicEmbeddingFunction:
        def __init__(self, *, dims: int = 96) -> None:
            self._dims = dims

        def __call__(self, input):  # noqa: ANN001,ANN201
            if isinstance(input, str):
                values = [input]
            else:
                values = [str(item) for item in input]
            vectors: list[list[float]] = []
            for value in values:
                digest = hashlib.sha256(value.encode("utf-8")).digest()
                vector = [((digest[idx % len(digest)] / 255.0) * 2.0) - 1.0 for idx in range(self._dims)]
                vectors.append(vector)
            return vectors

        def embed_query(self, input):  # noqa: ANN001,ANN201
            return self.__call__(input)

        @staticmethod
        def name() -> str:
            return "default"

        @staticmethod
        def build_from_config(config: dict[str, object]):  # noqa: ANN205
            dims_raw = config.get("dims")
            if isinstance(dims_raw, int) and dims_raw > 0:
                return _DeterministicEmbeddingFunction(dims=dims_raw)
            return _DeterministicEmbeddingFunction()

        def get_config(self) -> dict[str, object]:
            return {"dims": self._dims}

        def is_legacy(self) -> bool:
            return False

        def default_space(self) -> str:
            return "cosine"

        def supported_spaces(self) -> list[str]:
            return ["cosine", "l2", "ip"]

    monkeypatch.setattr(
        "dcode_app_factory.registry._build_embedding_function",
        lambda *, model_name: _DeterministicEmbeddingFunction(),
    )


def test_web_search_requires_brightdata_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRIGHTDATA_API_KEY", raising=False)
    monkeypatch.setenv("BRIGHTDATA_SERP_ZONE", "serp_zone")

    with pytest.raises(RuntimeError, match="BRIGHTDATA_API_KEY"):
        tools_module.web_search.func("latency optimization patterns")


def test_web_search_requires_brightdata_serp_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "token")
    monkeypatch.delenv("BRIGHTDATA_SERP_ZONE", raising=False)

    with pytest.raises(RuntimeError, match="BRIGHTDATA_SERP_ZONE"):
        tools_module.web_search.func("distributed caching benchmark")


def test_web_search_calls_brightdata_request_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "token")
    monkeypatch.setenv("BRIGHTDATA_SERP_ZONE", "serp_zone")
    monkeypatch.setenv("BRIGHTDATA_SERP_COUNTRY", "us")
    captured: dict[str, object] = {}

    def _fake_http_post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {
            "organic": [
                {
                    "link": "https://example.com/a",
                    "title": "A",
                    "description": "Result A",
                },
                {
                    "url": "https://example.com/b",
                    "name": "B",
                    "snippet": "Result B",
                },
            ]
        }

    monkeypatch.setattr(tools_module, "_http_post_json", _fake_http_post_json)

    output = tools_module.web_search.func("market making guide")
    payload = json.loads(output)

    assert captured["url"] == "https://api.brightdata.com/request"
    assert captured["headers"] == {"Authorization": "Bearer token"}
    assert captured["payload"] == {
        "zone": "serp_zone",
        "url": "https://www.google.com/search?q=market+making+guide",
        "format": "json",
        "country": "us",
        "method": "GET",
    }
    assert payload == {
        "provider": "brightdata",
        "query": "market making guide",
        "results": [
            {
                "url": "https://example.com/a",
                "title": "A",
                "snippet": "Result A",
            },
            {
                "url": "https://example.com/b",
                "title": "B",
                "snippet": "Result B",
            },
        ],
    }


def test_web_search_http_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the HTTP request to Bright Data fails, the exception should propagate
    to the caller rather than returning a partial or empty result.
    """
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "token")
    monkeypatch.setenv("BRIGHTDATA_SERP_ZONE", "serp_zone")

    def _failing_http_post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(tools_module, "_http_post_json", _failing_http_post_json)

    with pytest.raises(urllib.error.URLError, match="Connection refused"):
        tools_module.web_search.func("failing search query")


def test_web_search_returns_empty_results_for_empty_organic(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the Bright Data response has no organic results, the tool returns
    an empty results list rather than crashing.
    """
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "token")
    monkeypatch.setenv("BRIGHTDATA_SERP_ZONE", "serp_zone")

    def _empty_response(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        return {"organic": []}

    monkeypatch.setattr(tools_module, "_http_post_json", _empty_response)

    output = tools_module.web_search.func("no results query")
    payload = json.loads(output)
    assert payload["results"] == []
    assert payload["query"] == "no results query"


def test_search_code_index_returns_ranked_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """search_code_index should return ranked JSON results from a real ChromaDB
    code index populated with a test contract.
    """
    settings = RuntimeSettings(
        state_store_root=str(tmp_path),
        project_id="ACME-TOOL-INDEX",
        embedding_model="text-embedding-3-large",
    )
    state_store = FactoryStateStore(tmp_path, project_id=settings.project_id)
    index = CodeIndex(state_store.code_index_dir, embedding_model=settings.embedding_model)
    contract = MicroModuleContract(
        module_id="MM-tool-search",
        module_version="1.0.0",
        name="Request Validator",
        purpose="Validate incoming API requests against schema constraints",
        tags=["validation", "api", "request"],
        examples_ref="/modules/MM-tool-search/1.0.0/examples.md",
        created_by="tester",
        inputs=[ContractInput(name="request", type="dict", constraints=["non-empty"])],
        outputs=[ContractOutput(name="validated", type="dict", invariants=["normalized"])],
        error_surfaces=[ContractErrorSurface(name="ValidationError", when="bad payload", surface="exception")],
        effects=[{"type": EffectType.WRITE, "target": "state_store", "description": "writes traces"}],
        modes=ContractModes(sync=True, **{"async": False}, notes="sync"),
        error_cases=["bad payload"],
        dependencies=[],
        compatibility=CompatibilityRule(backward_compatible_with=[], breaking_change_policy="major"),
        runtime_budgets=RuntimeBudgets(latency_ms_p95=10, memory_mb_max=32),
        status=ArtifactStatus.DRAFT,
    )
    index.register(contract)

    # Patch the environment and project root so search_code_index can find the index
    monkeypatch.setenv("FACTORY_STATE_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("FACTORY_PROJECT_ID", settings.project_id)
    monkeypatch.setenv("FACTORY_EMBEDDING_MODEL", settings.embedding_model)

    output = tools_module.search_code_index.func("API request validation")
    results = json.loads(output)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["entry"]["module_id"] == "MM-tool-search"
    assert isinstance(results[0]["similarity_score"], float)

