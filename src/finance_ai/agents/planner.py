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
DETAILED_KEYWORDS = {"detailed", "deep", "thorough", "full"}
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

INTENT_SECTIONS: dict[str, list[str]] = {
    "price": ["summary", "recommendation", "chart_data", "source_items", "decision_rationale"],
    "fundamentals": ["summary", "bull_case", "bear_case", "source_items", "decision_rationale"],
    "news": ["summary", "bull_case", "bear_case", "source_items", "decision_rationale"],
    "rag": ["summary", "bull_case", "bear_case", "source_items", "decision_rationale"],
    "compare": [
        "summary",
        "comparison_view",
        "recommendation",
        "bull_case",
        "bear_case",
        "source_items",
        "chart_data",
        "decision_rationale",
    ],
    "market_ideas": ["summary", "recommendation", "bull_case", "bear_case", "source_items", "decision_rationale"],
    "market_general": ["summary", "recommendation", "bull_case", "bear_case", "source_items", "decision_rationale"],
    "unknown": ["summary", "decision_rationale"],
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
    if len(entities) >= 2 and _contains_any(query_lower, COMPARE_QUALIFIERS):
        return True
    if " and " in query_lower and len(entities) >= 2:
        return True
    return False


def _build_execution_steps(tool_sequence: list[str]) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = [{"step": "plan", "action": "classify_intent_and_tools"}]
    for tool in tool_sequence:
        steps.append({"step": "act", "action": tool})
    steps.append({"step": "synthesize", "action": "build_structured_response"})
    return steps


def _response_sections(intent: str) -> list[str]:
    return INTENT_SECTIONS.get(intent, INTENT_SECTIONS["unknown"])


def _tool_choice_rationale(intent: str, tool_sequence: list[str]) -> str:
    if not tool_sequence:
        return "No tools planned; retrieving fallback answer."
    if intent == "compare":
        return f"Comparing two entities; executing for both: {', '.join(tool_sequence)}."
    if intent == "market_ideas":
        return f"Market-wide screening; using tools: {', '.join(tool_sequence)}."
    if intent == "market_general":
        return f"Broad macro/market query with no single ticker; using tools: {', '.join(tool_sequence)}."
    return f"Executing tools to answer query: {', '.join(tool_sequence)}."


def _fallback_tools_for_intent(intent: str) -> list[str]:
    fallbacks: dict[str, list[str]] = {
        "price": ["rag_retriever"],
        "fundamentals": ["rag_retriever", "search_news"],
        "news": ["rag_retriever"],
        "compare": ["rag_retriever"],
        "rag": ["search_news"],
        "market_ideas": ["rag_retriever"],
        "market_general": ["rag_retriever"],
    }
    return fallbacks.get(intent, [])


def _llm_plan(query: str, entities: list[ResolvedCompany]) -> QueryPlan | None:
    entity_payload = [{"ticker": item.ticker, "company_name": item.company_name} for item in entities]
    prompt = (
        "You are a financial query planner. Return ONLY valid JSON with keys: "
        "intent, is_comparison, requires_rag, requires_news, response_style, confidence_low, tool_sequence, reasoning, execution_steps, response_sections. "
        "intent must be one of: price,fundamentals,news,rag,compare,market_ideas,market_general,unknown. "
        "response_style must be short or detailed. "
        f"Query: {query}\n"
        f"Entities: {entity_payload}\n"
    )
    try:
        raw = invoke_ollama(prompt)
        payload = _extract_json_object(raw)
        if not payload:
            return None
        return QueryPlan.model_validate(payload)
    except Exception:
        return None


def deterministic_fallback_plan(query: str, entities: list[ResolvedCompany]) -> QueryPlan:
    query_lower = query.lower()
    has_news_keywords = _contains_any(query_lower, NEWS_KEYWORDS)
    has_rag_keywords = _contains_any(query_lower, RAG_KEYWORDS)
    has_fundamental_keywords = _contains_any(query_lower, FUNDAMENTAL_KEYWORDS)
    has_price_keywords = _contains_any(query_lower, PRICE_KEYWORDS)

    if _is_compare_query(query_lower, entities):
        plan = QueryPlan(
            intent="compare",
            is_comparison=True,
            requires_rag=False,
            requires_news=True,
            response_style="detailed",
            confidence_low=len(entities) < 2,
            tool_sequence=["get_stock_price", "get_fundamentals", "search_news"],
            reasoning="Comparison detected; always fetch price, fundamentals, and recent news.",
        )
        plan.execution_steps = _build_execution_steps(plan.tool_sequence)
        plan.response_sections = _response_sections(plan.intent)
        return plan
    if _contains_any(query_lower, BUY_SELL_KEYWORDS) or (
        _contains_any(query_lower, INDIA_MARKET_KEYWORDS) and not entities
    ):
        plan = QueryPlan(
            intent="market_ideas",
            requires_rag=True,
            requires_news=True,
            response_style="detailed",
            confidence_low=False,
            tool_sequence=["get_india_market_ideas", "search_news", "rag_retriever"],
            reasoning="Generic buy/sell or broad India market query detected.",
        )
        plan.execution_steps = _build_execution_steps(plan.tool_sequence)
        plan.response_sections = _response_sections(plan.intent)
        return plan

    if _contains_any(query_lower, MARKET_GENERAL_KEYWORDS) and not entities:
        use_india_market_tool = _contains_any(query_lower, INDIA_MARKET_KEYWORDS)
        tool_sequence = ["search_news"]
        if use_india_market_tool:
            tool_sequence.insert(0, "get_india_market_ideas")
        plan = QueryPlan(
            intent="market_general",
            requires_rag=False,
            requires_news=True,
            response_style="detailed",
            confidence_low=False,
            tool_sequence=tool_sequence,
            reasoning="Broad market query with no specific entity.",
        )
        plan.execution_steps = _build_execution_steps(plan.tool_sequence)
        plan.response_sections = _response_sections(plan.intent)
        return plan

    if has_fundamental_keywords:
        plan = QueryPlan(
            intent="fundamentals",
            tool_sequence=["get_fundamentals"],
            response_style="short",
            reasoning="Fundamental/valuation keyword detected.",
        )
        plan.execution_steps = _build_execution_steps(plan.tool_sequence)
        plan.response_sections = _response_sections(plan.intent)
        return plan
    if has_price_keywords:
        plan = QueryPlan(
            intent="price",
            tool_sequence=["get_stock_price"],
            response_style="short",
            confidence_low=not bool(entities),
            reasoning="Price/trend intent detected.",
        )
        plan.execution_steps = _build_execution_steps(plan.tool_sequence)
        plan.response_sections = _response_sections(plan.intent)
        return plan
    if has_news_keywords:
        plan = QueryPlan(
            intent="news",
            requires_rag=True,
            requires_news=True,
            response_style="short",
            tool_sequence=["search_news", "rag_retriever"],
            reasoning="News-focused question requires current headlines and context.",
        )
        plan.execution_steps = _build_execution_steps(plan.tool_sequence)
        plan.response_sections = _response_sections(plan.intent)
        return plan
    if has_rag_keywords:
        plan = QueryPlan(
            intent="rag",
            requires_rag=True,
            requires_news=False,
            response_style="detailed",
            tool_sequence=["rag_retriever"],
            reasoning="Analysis/risk request benefits from retrieval grounding.",
        )
        plan.execution_steps = _build_execution_steps(plan.tool_sequence)
        plan.response_sections = _response_sections(plan.intent)
        return plan
    if entities:
        plan = QueryPlan(
            intent="price",
            tool_sequence=["get_stock_price"],
            response_style="short",
            confidence_low=False,
            reasoning="Company mention found with no stronger intent signals.",
        )
        plan.execution_steps = _build_execution_steps(plan.tool_sequence)
        plan.response_sections = _response_sections(plan.intent)
        return plan

    plan = QueryPlan(
        intent="unknown",
        tool_sequence=[],
        confidence_low=True,
        reasoning="No finance entity or clear intent detected.",
    )
    plan.execution_steps = _build_execution_steps(plan.tool_sequence)
    plan.response_sections = _response_sections(plan.intent)
    return plan


def plan_query(
    query: str,
    entities: list[ResolvedCompany],
    planner_mode: str | None = None,
) -> QueryPlan:
    settings = get_settings()
    mode = (planner_mode or settings.agent_planner_mode).lower()
    base = deterministic_fallback_plan(query, entities)

    if mode in {"llm", "hybrid"}:
        llm_candidate = _llm_plan(query, entities)
        # Never let an LLM 'unknown' downgrade a good deterministic plan.
        if llm_candidate is not None and (llm_candidate.intent != "unknown" or base.intent == "unknown"):
            base = llm_candidate

    query_lower = query.lower()

    if _contains_any(query_lower, DETAILED_KEYWORDS):
        base.response_style = "detailed"

    if base.intent in {"price", "fundamentals"} and base.confidence_low:
        base.requires_rag = True
        if "rag_retriever" not in base.tool_sequence:
            base.tool_sequence.append("rag_retriever")

    if base.intent == "compare":
        base.is_comparison = True
        base.requires_news = True
        if "search_news" not in base.tool_sequence:
            base.tool_sequence.append("search_news")

    if len(query.split()) >= 14:
        base.response_style = "detailed"

    if not base.response_sections:
        base.response_sections = _response_sections(base.intent)
    if not base.execution_steps:
        base.execution_steps = _build_execution_steps(base.tool_sequence)

    base.tool_choice_rationale = _tool_choice_rationale(base.intent, base.tool_sequence)
    base.fallback_tools = _fallback_tools_for_intent(base.intent)
    base.planning_confidence = 0.75 if not base.confidence_low else 0.45

    return base
