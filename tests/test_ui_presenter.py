from finance_ai.ui.presenter import (
    build_contextual_suggestions,
    fmt,
    format_metric_value,
    pct,
    price_series_frame,
    recommendation_color,
    scalar_stock_metrics,
)


def test_scalar_stock_metrics_filters_complex_values() -> None:
    metrics = scalar_stock_metrics(
        {
            "current_price": 123.4,
            "ticker": "AAPL",
            "price_series": [{"date": "2026-01-01", "close": 123.4}],
        }
    )
    assert "current_price" in metrics
    assert "ticker" in metrics
    assert "price_series" not in metrics


def test_price_series_frame_builds_sorted_frame() -> None:
    frame = price_series_frame(
        {
            "price_series": [
                {"date": "2026-01-02", "close": 101.0, "volume": 1000},
                {"date": "2026-01-01", "close": 100.0, "volume": 900},
            ]
        }
    )
    assert not frame.empty
    assert frame.iloc[0]["close"] == 100.0


def test_recommendation_color_mapping() -> None:
    assert recommendation_color("BUY") == "green"
    assert recommendation_color("SELL") == "red"
    assert recommendation_color("HOLD") == "orange"
    assert recommendation_color("OTHER") == "gray"


def test_format_metric_value_us_price() -> None:
    assert format_metric_value("current_price", 123.456, "AAPL") == "$123.46"


def test_format_metric_value_india_price() -> None:
    assert format_metric_value("current_price", 123.456, "RELIANCE.NS") == "Rs 123.46"


def test_format_metric_value_percent_precision() -> None:
    assert format_metric_value("change_pct", 2.3456, "AAPL") == "2.35%"


def test_pct_formatter_whole_percent() -> None:
    assert pct(0.634) == "63%"


def test_fmt_formatter_with_fallback_and_prefix() -> None:
    assert fmt(None, prefix="$") == "-"
    assert fmt(123.456, prefix="$") == "$123.46"


def test_build_contextual_suggestions_india_market_ideas() -> None:
    suggestions = build_contextual_suggestions(
        query="What stocks should I buy in India now?",
        intent="market_ideas",
        subject="Indian market",
        risk_view=["Regulation can affect lending and infra names."],
        news_view=["Policy support is improving capex sentiment."],
        stock_view={"india_nifty_1m_pct": 1.2},
    )

    assert len(suggestions) == 6
    assert any("india" in item.lower() or "nifty" in item.lower() for item in suggestions)
    assert any("regulatory" in item.lower() or "risk" in item.lower() for item in suggestions)


def test_build_contextual_suggestions_price_valuation_theme() -> None:
    suggestions = build_contextual_suggestions(
        query="Should I buy Apple after this rally?",
        intent="price",
        subject="Apple",
        risk_view=["Valuation is rich with elevated PE versus history."],
        news_view=["Analysts discuss whether the premium multiple can hold."],
        stock_view={"current_price": 190.2},
    )

    assert len(suggestions) == 6
    assert any("valuation" in item.lower() or "overvalued" in item.lower() for item in suggestions)
    assert any("apple" in item.lower() for item in suggestions)
