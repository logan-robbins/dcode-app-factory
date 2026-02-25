from __future__ import annotations

import json
import urllib.error

import pytest

from dcode_app_factory import tools as tools_module


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

