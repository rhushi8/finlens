from __future__ import annotations

from typing import Any

import pandas as pd


def scalar_stock_metrics(stock_view: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stock_view.items()
        if isinstance(value, (int, float, str)) or value is None
    }


def price_series_frame(stock_view: dict[str, Any]) -> pd.DataFrame:
    series = stock_view.get("price_series")
    if not isinstance(series, list) or not series:
        return pd.DataFrame(columns=["date", "close", "volume"])

    frame = pd.DataFrame(series)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
    return frame


def recommendation_color(recommendation: str) -> str:
    mapping = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
    return mapping.get((recommendation or "").upper(), "gray")


def pct(value: float) -> str:
    return f"{value:.0%}"


def fmt(
    value: float | None,
    template: str = "{:.2f}",
    prefix: str = "",
    suffix: str = "",
    fallback: str = "-",
) -> str:
    if value is None:
        return fallback
    return f"{prefix}{template.format(value)}{suffix}"


def is_indian_ticker(ticker: str | None) -> bool:
    if not ticker:
        return False
    upper = ticker.upper()
    return upper.endswith(".NS") or upper.endswith(".BO") or upper in {"^NSEI", "^BSESN", "^NSEBANK"}


def format_metric_value(metric_key: str, value: Any, ticker: str | None) -> str:
    if value is None:
        return "N/A"

    key = metric_key.lower()
    if isinstance(value, (int, float)):
        if "price" in key or key in {"period_high", "period_low"}:
            prefix = "Rs " if is_indian_ticker(ticker) else "$"
            return f"{prefix}{float(value):.2f}"
        if "pct" in key or "yield" in key:
            return f"{float(value):.2f}%"
        if "market_cap" in key:
            prefix = "Rs " if is_indian_ticker(ticker) else "$"
            return f"{prefix}{float(value):,.2f}"
        if isinstance(value, float):
            return f"{float(value):.2f}"
        return f"{int(value):,}"

    return str(value)


def build_contextual_suggestions(
    query: str,
    intent: str,
    subject: str,
    risk_view: list[str],
    news_view: list[str],
    stock_view: dict[str, Any],
) -> list[str]:
    q = (query or "").lower()
    text_blob = " ".join([query, *risk_view[:3], *news_view[:3]]).lower()
    subject_label = subject or "this company"

    suggestions: list[str] = [
        f"What changed recently for {subject_label}?",
        f"What are near-term risks for {subject_label}?",
        f"Summarize latest news affecting {subject_label}",
    ]

    if "supply" in text_blob or "supply chain" in text_blob:
        suggestions.append(f"How sensitive is {subject_label} to supply-chain disruptions?")
    if "valuation" in text_blob or "pe" in text_blob:
        suggestions.append(f"Is {subject_label} overvalued or undervalued versus peers?")
    if "earnings" in text_blob or "guidance" in text_blob:
        suggestions.append(f"How could the next earnings update impact {subject_label}?")
    if "regulation" in text_blob or "policy" in text_blob:
        suggestions.append(f"What regulatory risks matter most for {subject_label} now?")

    india_signals = {
        "india_universe_size",
        "india_risk_profile",
        "india_nifty_1m_pct",
        "india_sensex_1m_pct",
    }
    is_india_context = any(key in stock_view for key in india_signals) or any(
        token in q for token in ["india", "nifty", "sensex", "nse", "bse"]
    )

    if is_india_context:
        suggestions.extend(
            [
                "What sectors are leading in India this month?",
                "Compare NIFTY vs SENSEX trend and risk right now",
                "What stocks to buy in India for conservative investors?",
            ]
        )

    if intent == "price":
        suggestions.extend(
            [
                f"Show valuation metrics for {subject_label}",
                f"Compare {subject_label} with its closest peer",
            ]
        )
    elif intent == "fundamentals":
        suggestions.extend(
            [
                f"Give bullish vs bearish case for {subject_label}",
                f"What should I monitor next for {subject_label}?",
            ]
        )
    elif intent == "news":
        suggestions.extend(
            [
                f"How could these headlines impact {subject_label} price?",
                f"What is the 1-month setup for {subject_label} after this news?",
            ]
        )
    elif intent == "market_ideas":
        suggestions.extend(
            [
                "What stocks to avoid in India right now and why?",
                "Build a watchlist for India large-caps with low risk",
            ]
        )

    seen: set[str] = set()
    unique: list[str] = []
    for suggestion in suggestions:
        clean = suggestion.strip()
        if clean and clean not in seen:
            seen.add(clean)
            unique.append(clean)

    return unique[:6]
