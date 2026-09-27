import pandas as pd
import pytest

from finance_ai.tools import (
    calculate_financial_metric,
    get_fundamentals,
    get_india_market_ideas,
    get_stock_price,
    search_news,
)
from finance_ai.schemas.tools import StockPriceResponse


class TestStockTools:
    @pytest.fixture(autouse=True)
    def _mock_yfinance(self, monkeypatch):
        import finance_ai.tools.stock as stock_module

        index = pd.date_range("2026-01-01", periods=5, freq="D")
        hist = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 101.5, 103.0],
                "High": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Low": [99.0, 100.0, 101.0, 100.5, 102.0],
                "Close": [100.5, 101.5, 102.5, 103.0, 104.0],
                "Volume": [1000, 1200, 1300, 1250, 1400],
            },
            index=index,
        )

        def fake_download(ticker, period="1mo", progress=False, session=None):
            if ticker == "INVALID123456":
                return pd.DataFrame()
            return hist

        class FakeTicker:
            def __init__(self, symbol, session=None):
                self.symbol = symbol

            @property
            def info(self):
                if self.symbol == "FAKETTTT":
                    return {}
                return {
                    "longName": "Apple Inc.",
                    "sector": "Technology",
                    "marketCap": 2800000000000,
                    "trailingPE": 28.3,
                    "dividendYield": 0.005,
                    "fiftyTwoWeekHigh": 230.0,
                    "fiftyTwoWeekLow": 150.0,
                    "averageVolume": 52000000,
                    "beta": 1.2,
                }

        monkeypatch.setattr(stock_module.yf, "download", fake_download)
        monkeypatch.setattr(stock_module.yf, "Ticker", FakeTicker)

    def test_get_stock_price_valid_ticker(self) -> None:
        response = get_stock_price("AAPL", period="1mo")
        assert response.ticker == "AAPL"
        assert response.period == "1mo"
        assert response.current_price > 0
        assert response.error is None
        assert response.data_points is not None
        assert len(response.data_points) > 0

    def test_get_stock_price_invalid_ticker(self) -> None:
        response = get_stock_price("INVALID123456", period="1mo")
        assert response.ticker == "INVALID123456"
        assert response.error is not None

    def test_get_stock_price_case_insensitive(self) -> None:
        response = get_stock_price("aapl", period="1mo")
        assert response.ticker == "AAPL"

    def test_get_stock_price_indian_ticker_format(self) -> None:
        response = get_stock_price("RELIANCE.NS", period="1mo")
        assert response.ticker == "RELIANCE.NS"
        assert response.error is None

    def test_get_stock_price_change_calculation(self) -> None:
        response = get_stock_price("MSFT", period="1mo")
        if response.error is None and len(response.data_points) > 0:
            first_price = response.data_points[0].close
            last_price = response.current_price
            expected_change_abs = last_price - first_price
            assert abs(response.change_abs - expected_change_abs) < 1.0

    def test_get_fundamentals_valid_ticker(self) -> None:
        response = get_fundamentals("AAPL")
        assert response.ticker == "AAPL"
        assert response.error is None
        assert response.company_name is not None or response.sector is not None

    def test_get_fundamentals_invalid_ticker(self) -> None:
        response = get_fundamentals("FAKETTTT")
        assert response.ticker == "FAKETTTT"
        assert response.error is None or response.company_name is None


class TestCalculator:
    def test_pct_change_positive(self) -> None:
        response = calculate_financial_metric(
            "pct_change", {"old_value": 100, "new_value": 120}
        )
        assert response.result == 20.0
        assert response.formatted_result == "+20.00%"
        assert response.error is None

    def test_pct_change_negative(self) -> None:
        response = calculate_financial_metric(
            "pct_change", {"old_value": 100, "new_value": 80}
        )
        assert response.result == -20.0
        assert response.formatted_result == "-20.00%"
        assert response.error is None

    def test_pct_change_zero_old_value(self) -> None:
        response = calculate_financial_metric(
            "pct_change", {"old_value": 0, "new_value": 100}
        )
        assert response.error is not None

    def test_pe_multiple_calculation(self) -> None:
        response = calculate_financial_metric(
            "pe_multiple", {"earnings_per_share": 5.0, "target_pe": 20}
        )
        assert response.result == 100.0
        assert response.formatted_result == "$100.00"
        assert response.error is None

    def test_dividend_income_calculation(self) -> None:
        response = calculate_financial_metric(
            "dividend_income",
            {"shares_owned": 100, "annual_dividend_per_share": 2.5},
        )
        assert response.result == 250.0
        assert response.formatted_result == "$250.00"
        assert response.error is None

    def test_price_target_calculation(self) -> None:
        response = calculate_financial_metric(
            "price_target",
            {"current_price": 100, "target_price": 120},
        )
        assert response.result == 20.0
        assert response.formatted_result == "+20.00%"
        assert response.details["absolute_change"] == 20.0
        assert response.error is None

    def test_unknown_operation(self) -> None:
        response = calculate_financial_metric(
            "unknown_op", {"param1": 1, "param2": 2}
        )
        assert response.error is not None
        assert "Unknown operation" in response.error


class TestNews:
    @pytest.fixture(autouse=True)
    def _mock_requests(self, monkeypatch):
        import finance_ai.tools.news as news_module

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, params, timeout):
            payload = {
                "articles": [
                    {
                        "url": "https://example.com/apple",
                        "title": "Apple launches new chip",
                        "domain": "example.com",
                        "sourcecountry": "US",
                        "seendate": "20260406T120000Z",
                        "tone": "1.1",
                    },
                    {
                        "url": "https://example.com/tesla",
                        "title": "Tesla updates guidance",
                        "domain": "example.com",
                        "sourcecountry": "US",
                        "seendate": "20260406T130000Z",
                    },
                ]
            }
            return FakeResponse(payload)

        monkeypatch.setattr(news_module.requests, "get", fake_get)

    def test_search_news_valid_query(self) -> None:
        response = search_news("Apple")
        assert response.query == "Apple"
        assert isinstance(response.articles, list)

    def test_search_news_empty_query(self) -> None:
        response = search_news("")
        assert response.error is not None

    def test_search_news_max_results(self) -> None:
        response = search_news("Tesla", max_results=3)
        assert len(response.articles) <= 3


class TestIndiaMarketIdeas:
    @pytest.fixture(autouse=True)
    def _mock_stock_price(self, monkeypatch):
        import finance_ai.tools.india_market as india_market_module

        base_changes = {
            "RELIANCE.NS": 4.2,
            "TCS.NS": 1.1,
            "INFY.NS": -2.0,
            "HDFCBANK.NS": 0.8,
            "ICICIBANK.NS": 3.4,
            "SBIN.NS": -4.8,
            "^NSEI": 2.2,
            "^BSESN": 1.7,
            "^NSEBANK": 1.1,
        }

        def fake_get_stock_price(ticker: str, period: str = "1mo"):
            change = base_changes.get(ticker, 0.5)
            return StockPriceResponse(
                ticker=ticker,
                period=period,
                current_price=100.0,
                change_pct=change,
                change_abs=change,
                period_high=110.0,
                period_low=90.0,
                volume_avg=100000,
                data_points=None,
                retrieved_at=pd.Timestamp("2026-04-07"),
                error=None,
            )

        monkeypatch.setattr(india_market_module, "get_stock_price", fake_get_stock_price)

    def test_get_india_market_ideas_basic(self) -> None:
        response = get_india_market_ideas("What stocks to buy in India right now?")
        assert response.error is None
        assert len(response.ideas) > 0
        assert response.market_snapshot["risk_profile"] in {"conservative", "moderate", "aggressive"}
        assert "market_breadth" in response.market_snapshot
        assert "index_snapshot_1m_pct" in response.market_snapshot

    def test_get_india_market_ideas_action_present(self) -> None:
        response = get_india_market_ideas("aggressive India momentum stocks", max_results=3)
        assert len(response.ideas) <= 3
        assert all(idea.action in {"BUY", "HOLD", "WATCH", "SELL", "AVOID"} for idea in response.ideas)
        assert all(isinstance(idea.index_membership, list) for idea in response.ideas)
        assert all(idea.sector is not None for idea in response.ideas)
