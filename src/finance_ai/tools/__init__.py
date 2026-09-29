"""Lazy imports. Provider deps load only when a tool runs."""

from __future__ import annotations

from typing import Any

from finance_ai.tools.calculator import calculate_financial_metric


def get_stock_price(*args: Any, **kwargs: Any):
    from finance_ai.tools.stock import get_stock_price as _get_stock_price

    return _get_stock_price(*args, **kwargs)


def get_fundamentals(*args: Any, **kwargs: Any):
    from finance_ai.tools.stock import get_fundamentals as _get_fundamentals

    return _get_fundamentals(*args, **kwargs)


def get_india_market_ideas(*args: Any, **kwargs: Any):
    from finance_ai.tools.india_market import get_india_market_ideas as _get_india_market_ideas

    return _get_india_market_ideas(*args, **kwargs)


def search_news(*args: Any, **kwargs: Any):
    from finance_ai.tools.news import search_news as _search_news

    return _search_news(*args, **kwargs)


__all__ = [
    "get_stock_price",
    "get_fundamentals",
    "get_india_market_ideas",
    "calculate_financial_metric",
    "search_news",
]
