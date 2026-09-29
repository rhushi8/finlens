from __future__ import annotations

import json
import re

from finance_ai.config import get_settings
from finance_ai.llm.ollama_client import invoke_ollama
from finance_ai.schemas.agent import QueryPlan
from finance_ai.utils.company_resolution import ResolvedCompany

PRICE_KEYWORDS = {"price", "trend", "performance", "move", "chart", "up", "down"}
FUNDAMENTAL_KEYWORDS = {"fundamental", "valuation", "earnings", "dividend", "beta", "market cap", "pe"}
RAG_KEYWORDS = {"risk", "summarize", "summary", "why", "explain", "analysis", "outlook", "catalyst"}
NEWS_KEYWORDS = {"news", "headline", "headlines"}
COMPARE_KEYWORDS = {"compare", "versus", "vs"}
COMPARE_QUALIFIERS = {"better", "stronger", "safer", "cheaper", "weaker", "preferred"}
INDIA_MARKET_KEYWORDS = {
    "india",
    "indian",
    "nse",
    "bse",
    "sensex",
    "nifty",
    "bank nifty",
    "fii",
    "dii",
}
BUY_SELL_KEYWORDS = {
    "what to buy",
    "what to sell",
    "buy now",
    "sell now",
    "stocks to buy",
    "stocks to sell",
    "investment ideas",
}
KNOWN_TOOLS = {"get_stock_price", "get_fundamentals", "search_news", "get_india_market_ideas", "rag_retriever"}
MARKET_GENERAL_KEYWORDS = {
    "market",
    "global",
    "macro",
    "sector",
    "economy",
    "interest rate",
    "inflation",
    "fed",
    "rbi",
}


def _contains_any(query_lower: str, keywords: set[str]) -> bool:
    return any(keyword in query_lower for keyword in keywords)


def _extract_json_object(text: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _is_compare_query(query_lower: str, entities: list[ResolvedCompany]) -> bool:
    if _contains_any(query_lower, COMPARE_KEYWORDS):
        return True
    return len(entities) >= 2 and (_contains_any(query_lower, COMPARE_QUALIFIERS) or " and " in query_lower)


def _llm_plan(query: str, entities: list[ResolvedCompany]) -> QueryPlan | None:
    entity_payload = [{"ticker": item.ticker, "company_name": item.company_name} for item in entities]
    prompt = (
        "You are a financial query planner. Return ONLY valid JSON with keys: "
        "intent, requires_rag, requires_news, confidence_low, tool_sequence, reasoning. "
        "intent must be one of: price,fundamentals,news,rag,compare,market_ideas,market_general,unknown. "
        f"tool_sequence may only use: {','.join(sorted(KNOWN_TOOLS))}. "
        f"Query: {query}\n"
        f"Entities: {entity_payload}\n"
    )
    try:
        payload = _extract_json_object(invoke_ollama(prompt))
        if not payload:
            return None
        return QueryPlan.model_validate(payload)
    except Exception:
        return None


def deterministic_fallback_plan(query: str, entities: list[ResolvedCompany]) -> QueryPlan:
    query_lower = query.lower()

    if _is_compare_query(query_lower, entities):
        return QueryPlan(
            intent="compare",
            requires_news=True,
            confidence_low=len(entities) < 2,
            tool_sequence=["get_stock_price", "get_fundamentals", "search_news"],
            reasoning="Comparison detected; always fetch price, fundamentals, and recent news.",
        )
    if _contains_any(query_lower, BUY_SELL_KEYWORDS) or (
        _contains_any(query_lower, INDIA_MARKET_KEYWORDS) and not entities
    ):
        return QueryPlan(
            intent="market_ideas",
            requires_rag=True,
            requires_news=True,
            tool_sequence=["get_india_market_ideas", "search_news", "rag_retriever"],
            reasoning="Generic buy/sell or broad India market query detected.",
        )
    if _contains_any(query_lower, MARKET_GENERAL_KEYWORDS) and not entities:
        return QueryPlan(
            intent="market_general",
            requires_news=True,
            tool_sequence=["search_news"],
            reasoning="Broad market query with no specific entity.",
        )
    if _contains_any(query_lower, FUNDAMENTAL_KEYWORDS):
        return QueryPlan(
            intent="fundamentals",
            tool_sequence=["get_fundamentals"],
            reasoning="Fundamental/valuation keyword detected.",
        )
    if _contains_any(query_lower, PRICE_KEYWORDS):
        return QueryPlan(
            intent="price",
            tool_sequence=["get_stock_price"],
            confidence_low=not entities,
            reasoning="Price/trend intent detected.",
        )
    if _contains_any(query_lower, NEWS_KEYWORDS):
        return QueryPlan(
            intent="news",
            requires_rag=True,
            requires_news=True,
            tool_sequence=["search_news", "rag_retriever"],
            reasoning="News-focused question requires current headlines and context.",
        )
    if _contains_any(query_lower, RAG_KEYWORDS):
        return QueryPlan(
            intent="rag",
            requires_rag=True,
            tool_sequence=["rag_retriever"],
            reasoning="Analysis/risk request benefits from retrieval grounding.",
        )
    if entities:
        return QueryPlan(
            intent="price",
            tool_sequence=["get_stock_price"],
            reasoning="Company mention found with no stronger intent signals.",
        )
    return QueryPlan(
        intent="unknown",
        confidence_low=True,
        reasoning="No finance entity or clear intent detected.",
    )


def plan_query(
    query: str,
    entities: list[ResolvedCompany],
    planner_mode: str | None = None,
) -> QueryPlan:
    mode = (planner_mode or get_settings().agent_planner_mode).lower()
    plan = deterministic_fallback_plan(query, entities)

    # hybrid: rules first, the LLM only for queries they can't place. llm: LLM first.
    if mode == "llm" or (mode == "hybrid" and plan.intent == "unknown"):
        llm_plan = _llm_plan(query, entities)
        # The LLM may invent tool names; only take plans the router can actually run.
        if (
            llm_plan is not None
            and llm_plan.tool_sequence
            and set(llm_plan.tool_sequence) <= KNOWN_TOOLS
            and (llm_plan.intent != "unknown" or plan.intent == "unknown")
        ):
            plan = llm_plan

    if plan.intent in {"price", "fundamentals"} and plan.confidence_low:
        plan.requires_rag = True
        if "rag_retriever" not in plan.tool_sequence:
            plan.tool_sequence.append("rag_retriever")

    if plan.intent == "compare":
        plan.requires_news = True
        if "search_news" not in plan.tool_sequence:
            plan.tool_sequence.append("search_news")

    return plan
