from datetime import datetime

import pytest

from finance_ai.agents import extract_ticker, route_query
from finance_ai.schemas.rag import RetrievalResponse, RetrievedChunk
from finance_ai.schemas.tools import (
    FundamentalsResponse,
    IndiaMarketIdeasResponse,
    MarketIdea,
    NewsArticle,
    NewsSearchResponse,
    StockPriceResponse,
)


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int = 4, ticker: str | None = None, date_from=None, date_to=None):
        return RetrievalResponse(
            query=query,
            results=[
                RetrievedChunk(
                    text="Apple faces supply-chain pressure but has resilient services growth.",
                    source="docs/example_filings/apple_10k.md",
                    title="Apple 10K",
                    score=0.12,
                    ticker=ticker,
                    source_type="filing",
                    document_date="2025-11-01",
                    source_url="https://example.com/apple-10k",
                    metadata={"ticker": ticker, "source_type": "filing"},
                )
            ],
        )


@pytest.fixture(autouse=True)
def _mock_agent_dependencies(monkeypatch):
    import finance_ai.agents.planner as planner
    import finance_ai.agents.router as router
    import finance_ai.agents.synthesis as synthesis

    # Keep these tests independent of whether Ollama happens to be running.
    def no_llm(*args, **kwargs):
        raise ConnectionError("LLM disabled in tests")

    monkeypatch.setattr(planner, "invoke_ollama", no_llm)
    monkeypatch.setattr(synthesis, "invoke_ollama", no_llm)

    def fake_price(ticker: str, period: str = "1mo"):
        return StockPriceResponse(
            ticker=ticker,
            period=period,
            current_price=210.0,
            change_pct=2.5,
            change_abs=5.0,
            period_high=215.0,
            period_low=190.0,
            volume_avg=1200000,
            retrieved_at=datetime.now(),
            data_points=None,
        )

    def fake_fundamentals(ticker: str):
        return FundamentalsResponse(
            ticker=ticker,
            company_name="Apple Inc." if ticker == "AAPL" else "Microsoft Corp.",
            sector="Technology",
            market_cap=2800000000000,
            pe_ratio=28.1,
            dividend_yield=0.005,
            beta=1.1,
            retrieved_at=datetime.now(),
        )

    def fake_news(query: str, max_results: int = 5):
        return NewsSearchResponse(
            query=query,
            articles=[
                NewsArticle(
                    source="example.com",
                    title="Nvidia announces AI platform expansion",
                    url="https://example.com/nvidia-news",
                )
            ],
            retrieved_at=datetime.now(),
        )

    monkeypatch.setattr(router, "get_stock_price", fake_price)
    monkeypatch.setattr(router, "get_fundamentals", fake_fundamentals)
    monkeypatch.setattr(router, "search_news", fake_news)
    monkeypatch.setattr(
        router,
        "get_india_market_ideas",
        lambda query, max_results=5: IndiaMarketIdeasResponse(
            query=query,
            market_snapshot={"universe_size": 10, "screened": 10, "risk_profile": "moderate"},
            ideas=[
                MarketIdea(
                    ticker="RELIANCE.NS",
                    company_name="Reliance Industries",
                    action="BUY",
                    score=0.73,
                    rationale="Positive momentum and liquidity",
                    change_pct_1m=4.2,
                )
            ],
            retrieved_at=datetime.now(),
        ),
    )
    monkeypatch.setattr(router, "get_finance_retriever", lambda: _FakeRetriever())


class TestTickerExtraction:
    def test_extract_ticker_from_query(self) -> None:
        assert extract_ticker("What is the price of AAPL today?") == "AAPL"

    def test_extract_ticker_no_match(self) -> None:
        assert extract_ticker("What is the stock market doing today?") is None


class TestRouteQuery:
    def test_route_query_unknown(self) -> None:
        answer = route_query("hello there")
        assert answer.intent == "unknown"
        assert answer.error is not None

    def test_route_query_price(self) -> None:
        answer = route_query("What is the price of AAPL?")
        assert answer.intent == "price"
        assert answer.stock_view.get("current_price") == 210.0
        assert answer.source_count > 0

    def test_route_query_price_today(self) -> None:
        answer = route_query("What is the price of AAPL today?")
        assert answer.intent == "price"
        assert answer.stock_view.get("current_price") == 210.0

    def test_route_query_fundamentals(self) -> None:
        answer = route_query("What is the PE ratio of Apple?")
        assert answer.intent == "fundamentals"
        assert "pe_ratio" in answer.stock_view

    def test_route_query_news(self) -> None:
        answer = route_query("Summarize the latest news about Nvidia")
        assert answer.intent == "news"
        assert len(answer.news_view) > 0
        assert len(answer.citations) > 0
        assert answer.citations[0].title == "Nvidia announces AI platform expansion"

    def test_route_query_risk_uses_rag(self) -> None:
        answer = route_query("What are the risks for Apple?")
        assert answer.intent == "rag"
        assert len(answer.risk_view) > 0
        assert answer.grounding_score >= 0.0

    def test_route_query_structured_fields(self) -> None:
        answer = route_query("Compare Apple and Microsoft")
        assert answer.summary != ""
        assert answer.recommendation in {"BUY", "HOLD", "SELL"}
        assert isinstance(answer.tool_calls, list)
        assert answer.latency_ms > 0
        assert answer.comparison_view is not None
        assert answer.comparison_view.left.ticker == "AAPL"
        assert answer.comparison_view.right.ticker == "MSFT"
        assert answer.comparison_view.winner in {"Apple Inc.", "Microsoft Corp.", "Balanced"}
        assert answer.comparison_view.recommendation in {"BUY", "HOLD", "SELL"}
        assert answer.decision_rationale != ""
        assert len(answer.bull_case) > 0
        compare_tools = [call.tool_name for call in answer.tool_calls]
        assert compare_tools.count("get_stock_price") == 2
        assert compare_tools.count("get_fundamentals") == 2
        assert compare_tools.count("search_news") == 2

    def test_route_query_india_market_ideas(self) -> None:
        answer = route_query("What stocks to buy in India right now?")
        assert answer.intent == "market_ideas"
        assert any("Top India candidate" in item for item in answer.trend_view)
        assert answer.source_count > 0
        assert "india_universe_size" in answer.stock_view
        assert "india_nifty_1m_pct" in answer.stock_view

    def test_route_query_market_general(self) -> None:
        answer = route_query("What's the outlook for global markets today?")
        assert answer.intent == "market_general"
        assert len(answer.news_view) > 0
        assert "india_universe_size" not in answer.stock_view
        assert answer.recommendation in {"BUY", "SELL", "HOLD"}
        assert answer.error is None

    def test_route_query_partial_tool_failure_warns_but_continues(self, monkeypatch) -> None:
        import finance_ai.agents.router as router

        def failing_fundamentals(ticker: str):
            return FundamentalsResponse(
                ticker=ticker,
                retrieved_at=datetime.now(),
                error="fundamentals unavailable",
            )

        monkeypatch.setattr(router, "get_fundamentals", failing_fundamentals)

        answer = route_query("Compare Apple and Microsoft")
        assert answer.summary
        assert answer.comparison_view is not None
        assert any("Partial data:" in warning for warning in answer.warnings)
