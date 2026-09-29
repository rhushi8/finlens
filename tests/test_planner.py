import pytest

from finance_ai.agents.planner import deterministic_fallback_plan, plan_query
from finance_ai.utils.company_resolution import resolve_company_entities


@pytest.mark.unit
def test_plan_query_news_with_rag() -> None:
    entities = resolve_company_entities("Summarize latest news about Nvidia")
    plan = plan_query("Summarize latest news about Nvidia", entities)
    assert plan.intent == "news"
    assert plan.requires_news is True
    assert plan.requires_rag is True


@pytest.mark.unit
def test_plan_query_compare_needs_two_entities() -> None:
    entities = resolve_company_entities("Compare Apple and Microsoft")
    plan = deterministic_fallback_plan("Compare Apple and Microsoft", entities)
    assert plan.intent == "compare"
    assert plan.requires_news is True
    assert "search_news" in plan.tool_sequence


@pytest.mark.unit
def test_plan_query_compare_handles_natural_language_phrasing() -> None:
    entities = resolve_company_entities("Which is better, Apple or Microsoft?")
    plan = deterministic_fallback_plan("Which is better, Apple or Microsoft?", entities)
    assert plan.intent == "compare"


def test_plan_query_unknown_low_confidence() -> None:
    plan = plan_query("hello", [])
    assert plan.intent == "unknown"
    assert plan.confidence_low is True


def test_plan_query_price_today_does_not_misclassify_as_news() -> None:
    entities = resolve_company_entities("What is the price of AAPL today?")
    plan = deterministic_fallback_plan("What is the price of AAPL today?", entities)
    assert plan.intent == "price"
    assert "get_stock_price" in plan.tool_sequence


def test_plan_query_llm_mode_parses_json(monkeypatch) -> None:
    import finance_ai.agents.planner as planner_module

    class _Settings:
        agent_planner_mode = "llm"

    monkeypatch.setattr(planner_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(
        planner_module,
        "invoke_ollama",
        lambda prompt: '{"intent":"news","is_comparison":false,"requires_rag":true,"requires_news":true,"response_style":"short","confidence_low":false,"tool_sequence":["search_news","rag_retriever"],"reasoning":"llm"}',
    )

    entities = resolve_company_entities("latest news on Apple")
    plan = plan_query("latest news on Apple", entities)
    assert plan.intent == "news"
    assert plan.requires_news is True


def test_plan_query_llm_fallback_on_invalid_json(monkeypatch) -> None:
    import finance_ai.agents.planner as planner_module

    class _Settings:
        agent_planner_mode = "llm"

    monkeypatch.setattr(planner_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(planner_module, "invoke_ollama", lambda prompt: "not-json")

    entities = resolve_company_entities("Compare Apple and Microsoft")
    plan = plan_query("Compare Apple and Microsoft", entities)
    assert plan.intent == "compare"


def test_plan_query_llm_compare_without_news_is_normalized(monkeypatch) -> None:
    import finance_ai.agents.planner as planner_module

    class _Settings:
        agent_planner_mode = "llm"

    monkeypatch.setattr(planner_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(
        planner_module,
        "invoke_ollama",
        lambda prompt: '{"intent":"compare","is_comparison":true,"requires_rag":false,"requires_news":false,"response_style":"detailed","confidence_low":false,"tool_sequence":["get_stock_price","get_fundamentals"],"reasoning":"llm compare without news"}',
    )

    entities = resolve_company_entities("Compare Apple and Microsoft")
    plan = plan_query("Compare Apple and Microsoft", entities)
    assert plan.intent == "compare"
    assert plan.requires_news is True
    assert "search_news" in plan.tool_sequence


def test_plan_query_market_ideas_for_india_buy_sell() -> None:
    entities = resolve_company_entities("what stocks to buy in india right now")
    plan = deterministic_fallback_plan("what stocks to buy in india right now", entities)
    assert plan.intent == "market_ideas"
    assert "get_india_market_ideas" in plan.tool_sequence


def test_plan_query_market_general_for_macro_without_entities() -> None:
    entities = resolve_company_entities("What's the outlook for global markets with inflation and fed risk?")
    plan = deterministic_fallback_plan("What's the outlook for global markets with inflation and fed risk?", entities)
    assert plan.intent == "market_general"
    assert plan.requires_news is True
    assert plan.tool_sequence == ["search_news"]


def test_plan_query_india_macro_without_entities_routes_market_ideas() -> None:
    entities = resolve_company_entities("What is the India macro outlook with inflation?")
    plan = deterministic_fallback_plan("What is the India macro outlook with inflation?", entities)
    assert plan.intent == "market_ideas"
    assert "get_india_market_ideas" in plan.tool_sequence
