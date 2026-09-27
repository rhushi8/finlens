"""Lazy imports so extract_ticker doesn't pull in the RAG stack."""

from __future__ import annotations

from typing import Any

from finance_ai.agents.planner import deterministic_fallback_plan, plan_query
from finance_ai.agents.quality import (
    assess_grounding,
    calibrate_confidence,
    enforce_grounded_wording,
)


def route_query(*args: Any, **kwargs: Any):
    from finance_ai.agents.router import route_query as _route_query

    return _route_query(*args, **kwargs)


def extract_ticker(*args: Any, **kwargs: Any):
    from finance_ai.agents.router import extract_ticker as _extract_ticker

    return _extract_ticker(*args, **kwargs)


__all__ = [
    "route_query",
    "extract_ticker",
    "plan_query",
    "deterministic_fallback_plan",
    "assess_grounding",
    "enforce_grounded_wording",
    "calibrate_confidence",
]
