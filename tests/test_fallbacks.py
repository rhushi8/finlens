"""Resilience tests: the system must degrade gracefully when external
services (GDELT, Ollama) are unavailable."""

from types import SimpleNamespace

import pytest

import finance_ai.tools.news as news_module
import finance_ai.agents.synthesis as synthesis_module
from finance_ai.agents.synthesis import synthesize_grounded_response
from finance_ai.tools.news import search_news


class _FakeTicker:
    def __init__(self, items):
        self.news = items


@pytest.fixture
def _force_gdelt_failure(monkeypatch):
    """Make the GDELT call raise so the yfinance fallback path is exercised."""
    def boom(*args, **kwargs):
        raise news_module.requests.RequestException("GDELT down")

    monkeypatch.setattr(news_module.requests, "get", boom)
    monkeypatch.setattr(
        news_module,
        "resolve_primary_company",
        lambda query: SimpleNamespace(ticker="AAPL", company_name="Apple Inc."),
    )


def test_news_fallback_old_yfinance_schema(_force_gdelt_failure, monkeypatch):
    items = [
        {
            "title": "Apple ships M5",
            "link": "https://example.com/a",
            "publisher": "Reuters",
            "providerPublishTime": 1_700_000_000,
        }
    ]
    monkeypatch.setattr(news_module.yf, "Ticker", lambda t: _FakeTicker(items))

    resp = search_news("Apple", max_results=3)
    assert resp.error is None
    assert len(resp.articles) == 1
    assert resp.articles[0].title == "Apple ships M5"
    assert "Reuters" in resp.articles[0].source


def test_news_fallback_new_nested_content_schema(_force_gdelt_failure, monkeypatch):
    # yfinance >= 0.2.40 nests fields under "content".
    items = [
        {
            "content": {
                "title": "Apple beats estimates",
                "canonicalUrl": {"url": "https://example.com/b"},
                "provider": {"displayName": "Bloomberg"},
                "pubDate": "2026-04-06T12:00:00Z",
            }
        }
    ]
    monkeypatch.setattr(news_module.yf, "Ticker", lambda t: _FakeTicker(items))

    resp = search_news("Apple", max_results=3)
    assert resp.error is None
    assert len(resp.articles) == 1
    assert resp.articles[0].title == "Apple beats estimates"
    assert resp.articles[0].url == "https://example.com/b"
    assert resp.articles[0].published_date is not None  # ISO pubDate parsed


def test_synthesis_falls_back_when_ollama_unavailable(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("ollama not running")

    monkeypatch.setattr(synthesis_module, "invoke_ollama", boom)

    summary, recommendation, rationale = synthesize_grounded_response(
        query="Is AAPL a buy?",
        subject="Apple Inc.",
        tool_summary="Apple moved +3% over 1mo.",
        evidence_lines=["Apple moved +3% over 1mo."],
    )
    assert recommendation == "HOLD"
    assert summary  # non-empty deterministic fallback thesis
    assert "fallback" in rationale.lower()
