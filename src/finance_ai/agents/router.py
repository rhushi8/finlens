from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Any

from finance_ai.agents.planner import deterministic_fallback_plan, plan_query
from finance_ai.agents.quality import (
    assess_grounding,
    calibrate_confidence,
    enforce_grounded_wording,
)
from finance_ai.agents.synthesis import synthesize_grounded_response
from finance_ai.config import get_settings
from finance_ai.rag.service import get_finance_retriever
from finance_ai.schemas.agent import AnalystAnswer, Citation, ComparisonLeg, ComparisonView, SourceItem
from finance_ai.schemas.tools import ToolTrace
from finance_ai.tools import get_fundamentals, get_india_market_ideas, get_stock_price, search_news
from finance_ai.utils.company_resolution import resolve_company_entities, resolve_primary_company

logger = logging.getLogger(__name__)


def _company_label(entity) -> str:
    if entity is None:
        return "Finance topic"
    return entity.company_name or entity.ticker


def _run_tool(tool_name: str, input_params: dict[str, Any], fn, *args, **kwargs) -> tuple[Any, ToolTrace]:
    start = time.time()
    result = fn(*args, **kwargs)
    duration_ms = (time.time() - start) * 1000
    error = getattr(result, "error", None)
    output = result.model_dump() if hasattr(result, "model_dump") else {"value": str(result)}
    trace = ToolTrace(
        tool_name=tool_name,
        input_params=input_params,
        output=output,
        duration_ms=duration_ms,
        success=error is None,
        error=error,
    )
    return result, trace


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _dedupe_source_items(items: list[SourceItem]) -> list[SourceItem]:
    unique: list[SourceItem] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.title, item.source, item.url or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _build_citations(source_items: list[SourceItem], ticker: str | None) -> list[Citation]:
    citations: list[Citation] = []
    for item in source_items:
        citations.append(
            Citation(
                title=item.title,
                source=item.source,
                url=item.url,
                ticker=ticker,
                source_type=item.source_type or "market_or_doc",
                published_at=item.date,
                snippet=item.snippet,
            )
        )
    return citations


def _query_relevance(query: str, evidence_lines: list[str]) -> float:
    query_terms = {term for term in query.lower().split() if len(term) > 2}
    if not query_terms or not evidence_lines:
        return 0.0
    evidence_text = " ".join(evidence_lines).lower()
    hit_count = sum(1 for term in query_terms if term in evidence_text)
    return min(1.0, hit_count / max(1, len(query_terms)))


def _extract_cases(trend_view: list[str], risk_view: list[str], news_view: list[str]) -> tuple[list[str], list[str]]:
    bull_case = [item for item in (trend_view + news_view) if item][:3]
    bear_case = [item for item in risk_view if item][:3]
    return bull_case, bear_case


def _comparison_bear_points(leg: ComparisonLeg) -> list[str]:
    points: list[str] = []
    if leg.change_pct_1m is not None and leg.change_pct_1m < 0:
        points.append(f"{leg.company_name} has negative 1M momentum at {leg.change_pct_1m:+.2f}%.")
    if leg.pe_ratio is not None and leg.pe_ratio > 30:
        points.append(f"{leg.company_name} trades at a rich valuation with a P/E of {leg.pe_ratio:.2f}.")
    if leg.beta is not None and leg.beta > 1.2:
        points.append(f"{leg.company_name} carries elevated volatility with beta {leg.beta:.2f}.")
    if not points:
        points.append(f"Monitor {leg.company_name} for earnings revisions and sentiment shifts.")
    return points[:3]


def _compute_comparison_decision(left: ComparisonLeg, right: ComparisonLeg) -> tuple[str, str, str, list[str]]:
    left_score = 0.0
    right_score = 0.0
    total_weight = 0.0
    deltas: list[str] = []

    metrics: list[tuple[str, float, bool, str]] = [
        ("change_pct_1m", 1.0, True, "1M momentum"),
        ("pe_ratio", 0.75, False, "Valuation (P/E)"),
        ("beta", 0.5, False, "Volatility (beta)"),
        ("dividend_yield", 0.4, True, "Dividend yield"),
        ("market_cap", 0.25, True, "Market cap"),
    ]

    for attr, weight, higher_is_better, label in metrics:
        left_value = getattr(left, attr, None)
        right_value = getattr(right, attr, None)
        if left_value is None or right_value is None:
            continue

        total_weight += weight
        if left_value != right_value:
            left_better = left_value > right_value if higher_is_better else left_value < right_value
            if left_better:
                left_score += weight
            else:
                right_score += weight

        if attr == "dividend_yield":
            deltas.append(f"{label}: {left_value * 100:.2f}% vs {right_value * 100:.2f}%.")
        elif attr == "market_cap":
            deltas.append(f"{label}: {left_value:,.0f} vs {right_value:,.0f}.")
        elif attr == "change_pct_1m":
            deltas.append(
                f"{label}: {left.company_name} {left_value:+.2f}% vs {right.company_name} {right_value:+.2f}%."
            )
        else:
            deltas.append(f"{label}: {left_value:.2f} vs {right_value:.2f}.")

    if total_weight == 0:
        return "Balanced", "HOLD", "Not enough overlapping comparable metrics to determine a winner.", deltas

    left_norm = left_score / total_weight
    right_norm = right_score / total_weight
    gap = left_norm - right_norm

    if gap > 0.14:
        return left.company_name, "BUY", f"{left.company_name} scores higher across available comparison metrics.", deltas
    if gap < -0.14:
        return right.company_name, "BUY", f"{right.company_name} scores higher across available comparison metrics.", deltas
    return "Balanced", "HOLD", "Signals are mixed; no clear winner after risk/valuation balancing.", deltas


def _run_company_snapshot(
    *,
    entity,
    need_price: bool,
    need_fundamentals: bool,
    need_news: bool,
) -> tuple[dict[str, Any], list[ToolTrace], list[str], list[str], list[str], list[str], list[SourceItem]]:
    ticker = entity.ticker
    company_name = entity.company_name

    stock_view: dict[str, Any] = {}
    trend_view: list[str] = []
    news_view: list[str] = []
    warnings: list[str] = []
    evidence_lines: list[str] = []
    tool_calls: list[ToolTrace] = []
    source_items: list[SourceItem] = []

    tasks: list[tuple[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        if need_price:
            price_future = executor.submit(
                _run_tool,
                "get_stock_price",
                {"ticker": ticker, "period": "1mo"},
                get_stock_price,
                ticker,
                period="1mo",
            )
            tasks.append(("price", price_future))

        if need_fundamentals:
            fundamentals_future = executor.submit(
                _run_tool,
                "get_fundamentals",
                {"ticker": ticker},
                get_fundamentals,
                ticker,
            )
            tasks.append(("fundamentals", fundamentals_future))

        if need_news:
            news_future = executor.submit(
                _run_tool,
                "search_news",
                {"query": company_name or ticker, "max_results": 3},
                search_news,
                company_name or ticker,
                max_results=3,
            )
            tasks.append(("news", news_future))

        completed: dict[str, tuple[Any, ToolTrace]] = {}
        future_to_task = {future: task_name for task_name, future in tasks}
        for future in as_completed(future_to_task):
            completed[future_to_task[future]] = future.result()

    if "price" in completed:
        price_resp, trace = completed["price"]
        tool_calls.append(trace)
        if price_resp.error:
            warnings.append(price_resp.error)
        else:
            stock_view.update(
                {
                    "current_price": price_resp.current_price,
                    "change_pct": price_resp.change_pct,
                    "period_high": price_resp.period_high,
                    "period_low": price_resp.period_low,
                    "avg_volume": price_resp.volume_avg,
                }
            )
            if price_resp.data_points:
                stock_view["price_series"] = [
                    {
                        "date": item.date.isoformat(),
                        "close": item.close,
                        "volume": item.volume,
                    }
                    for item in price_resp.data_points[-60:]
                ]
            line = f"{company_name} moved {price_resp.change_pct:+.2f}% over {price_resp.period}."
            trend_view.append(line)
            evidence_lines.append(line)
            source_label = f"yfinance: {ticker} {price_resp.period} OHLCV"
            source_items.append(
                SourceItem(
                    title=f"{company_name} price history",
                    source_type="market_data",
                    source=source_label,
                    snippet=line,
                )
            )

    if "fundamentals" in completed:
        fund_resp, trace = completed["fundamentals"]
        tool_calls.append(trace)
        if fund_resp.error:
            warnings.append(fund_resp.error)
        else:
            stock_view.update(
                {
                    "market_cap": fund_resp.market_cap,
                    "pe_ratio": fund_resp.pe_ratio,
                    "dividend_yield": fund_resp.dividend_yield,
                    "beta": fund_resp.beta,
                }
            )
            line = f"{company_name} fundamentals: market cap {fund_resp.market_cap}, P/E {fund_resp.pe_ratio}, beta {fund_resp.beta}."
            trend_view.append(line)
            evidence_lines.append(line)
            source_label = f"yfinance: {ticker} fundamentals"
            source_items.append(
                SourceItem(
                    title=f"{company_name} fundamentals",
                    source_type="fundamentals",
                    source=source_label,
                    snippet=line,
                )
            )

    if "news" in completed:
        news_resp, trace = completed["news"]
        tool_calls.append(trace)
        if news_resp.error:
            warnings.append(news_resp.error)
        for article in news_resp.articles:
            bullet = f"{article.title} ({article.source})"
            news_view.append(bullet)
            evidence_lines.append(bullet)
            source_items.append(
                SourceItem(
                    title=article.title,
                    source_type="news",
                    source=article.source,
                    url=article.url,
                    date=article.published_date,
                    snippet=article.summary,
                )
            )

    return stock_view, tool_calls, trend_view, news_view, warnings, evidence_lines, source_items


def _run_compare_snapshots_parallel(
    *,
    left_entity,
    right_entity,
    need_price: bool,
    need_fundamentals: bool,
    need_news: bool,
):
    with ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(
            _run_company_snapshot,
            entity=left_entity,
            need_price=need_price,
            need_fundamentals=need_fundamentals,
            need_news=need_news,
        )
        right_future = executor.submit(
            _run_company_snapshot,
            entity=right_entity,
            need_price=need_price,
            need_fundamentals=need_fundamentals,
            need_news=need_news,
        )

        return left_future.result(), right_future.result()


def route_query(
    query: str,
    model_name: str | None = None,
    planner_mode: str | None = None,
) -> AnalystAnswer:
    """query -> planner -> tools -> retrieval -> synthesis."""

    started = time.time()
    settings = get_settings()

    entities = resolve_company_entities(query)
    primary = entities[0] if entities else resolve_primary_company(query)
    comparison_entities = entities[:2] if len(entities) >= 2 else []
    ticker = primary.ticker if primary else None
    company_name = primary.company_name if primary else None

    plan = plan_query(query, entities, planner_mode=planner_mode)
    if plan.intent == "unknown":
        plan = deterministic_fallback_plan(query, entities)

    tool_calls: list[ToolTrace] = []
    warnings: list[str] = []
    errors: list[str] = []
    sources: list[str] = []
    evidence_lines: list[str] = []
    stock_view: dict[str, Any] = {}
    news_view: list[str] = []
    risk_view: list[str] = []
    trend_view: list[str] = []
    source_items: list[SourceItem] = []
    comparison_view: ComparisonView | None = None
    bull_case: list[str] = []
    bear_case: list[str] = []
    decision_rationale = ""
    chart_data: dict[str, Any] = {}
    recommendation = "HOLD"
    summary: str = ""

    if plan.intent == "unknown":
        latency_ms = (time.time() - started) * 1000
        answer = AnalystAnswer(
            query=query,
            ticker=ticker,
            company_name=company_name,
            intent="unknown",
            summary="Unable to identify a stock or company in your query. Mention a ticker (AAPL, MSFT) or company name.",
            recommendation="HOLD",
            recommendation_confidence=0.0,
            grounding_score=0.0,
            tool_calls=[],
            citations=[],
            source_count=0,
            latency_ms=latency_ms,
            warnings=["No clear entity detected."],
            error="No ticker found",
        )
        return answer

    try:
        if plan.intent == "compare" and len(comparison_entities) >= 2:
            left_entity, right_entity = comparison_entities
            need_price = "get_stock_price" in plan.tool_sequence
            need_fundamentals = "get_fundamentals" in plan.tool_sequence
            need_news = "search_news" in plan.tool_sequence or plan.requires_news

            (
                left_stock,
                left_calls,
                left_trend,
                left_news,
                left_warnings,
                left_evidence,
                left_source_items,
            ), (
                right_stock,
                right_calls,
                right_trend,
                right_news,
                right_warnings,
                right_evidence,
                right_source_items,
            ) = _run_compare_snapshots_parallel(
                left_entity=left_entity,
                right_entity=right_entity,
                need_price=need_price,
                need_fundamentals=need_fundamentals,
                need_news=need_news,
            )

            tool_calls.extend(left_calls + right_calls)
            warnings.extend(left_warnings + right_warnings)
            trend_view.extend(left_trend + right_trend)
            news_view.extend(left_news + right_news)
            evidence_lines.extend(left_evidence + right_evidence)
            source_items.extend(left_source_items + right_source_items)

            left_leg = ComparisonLeg(
                ticker=left_entity.ticker,
                company_name=left_entity.company_name,
                price=left_stock.get("current_price"),
                change_pct_1m=left_stock.get("change_pct"),
                pe_ratio=left_stock.get("pe_ratio"),
                market_cap=left_stock.get("market_cap"),
                dividend_yield=left_stock.get("dividend_yield"),
                beta=left_stock.get("beta"),
                news_highlights=left_news[:2],
                bull_points=left_trend[:2],
            )
            left_leg.bear_points = _comparison_bear_points(left_leg)
            right_leg = ComparisonLeg(
                ticker=right_entity.ticker,
                company_name=right_entity.company_name,
                price=right_stock.get("current_price"),
                change_pct_1m=right_stock.get("change_pct"),
                pe_ratio=right_stock.get("pe_ratio"),
                market_cap=right_stock.get("market_cap"),
                dividend_yield=right_stock.get("dividend_yield"),
                beta=right_stock.get("beta"),
                news_highlights=right_news[:2],
                bull_points=right_trend[:2],
            )
            right_leg.bear_points = _comparison_bear_points(right_leg)

            winner_name, comparison_recommendation, decision_rationale, key_differences = _compute_comparison_decision(left_leg, right_leg)
            comparison_view = ComparisonView(
                left=left_leg,
                right=right_leg,
                winner=winner_name,
                recommendation=comparison_recommendation,
                winner_reason=decision_rationale,
                key_differences=key_differences,
            )
            summary, model_recommendation, model_rationale = synthesize_grounded_response(
                query=query,
                subject=f"{left_entity.company_name} vs {right_entity.company_name}",
                tool_summary="\n".join(trend_view + news_view),
                evidence_lines=evidence_lines,
                model_name=model_name,
                is_comparison=True,
            )
            stock_view = {
                "left": left_stock,
                "right": right_stock,
            }
            chart_data = {
                "comparison_price_change_pct": {
                    left_entity.ticker: left_stock.get("change_pct"),
                    right_entity.ticker: right_stock.get("change_pct"),
                },
                "comparison_pe_ratio": {
                    left_entity.ticker: left_stock.get("pe_ratio"),
                    right_entity.ticker: right_stock.get("pe_ratio"),
                },
            }
            sources.extend([item.source for item in source_items if item.source])
            bull_case = _dedupe(left_leg.bull_points + right_leg.bull_points)[:4]
            bear_case = _dedupe(left_leg.bear_points + right_leg.bear_points)[:4]
            if comparison_recommendation == "HOLD" and model_recommendation in {"BUY", "SELL", "HOLD"}:
                comparison_recommendation = model_recommendation
            recommendation = comparison_recommendation
            if not decision_rationale:
                decision_rationale = model_rationale
        elif plan.intent in {"price", "compare"} and ticker and "get_stock_price" in plan.tool_sequence:
            price_resp, trace = _run_tool(
                "get_stock_price",
                {"ticker": ticker, "period": "1mo"},
                get_stock_price,
                ticker,
                period="1mo",
            )
            tool_calls.append(trace)
            if price_resp.error:
                warnings.append(price_resp.error)
                errors.append(price_resp.error)
            else:
                stock_view.update(
                    {
                        "current_price": price_resp.current_price,
                        "change_pct": price_resp.change_pct,
                        "period_high": price_resp.period_high,
                        "period_low": price_resp.period_low,
                        "avg_volume": price_resp.volume_avg,
                    }
                )
                if price_resp.data_points:
                    stock_view["price_series"] = [
                        {
                            "date": item.date.isoformat(),
                            "close": item.close,
                            "volume": item.volume,
                        }
                        for item in price_resp.data_points[-60:]
                    ]
                trend_view.append(
                    f"{_company_label(primary)} moved {price_resp.change_pct:+.2f}% over {price_resp.period}."
                )
                evidence_lines.append(trend_view[-1])
                sources.append(f"yfinance: {ticker} {price_resp.period} OHLCV")
                source_items.append(
                    SourceItem(
                        title=f"{_company_label(primary)} price history",
                        source_type="market_data",
                        source=f"yfinance: {ticker} {price_resp.period} OHLCV",
                        snippet=trend_view[-1],
                    )
                )
                chart_data["price_series"] = stock_view.get("price_series", [])

        if plan.intent == "fundamentals" and ticker and "get_fundamentals" in plan.tool_sequence:
            fund_resp, trace = _run_tool(
                "get_fundamentals",
                {"ticker": ticker},
                get_fundamentals,
                ticker,
            )
            tool_calls.append(trace)
            if fund_resp.error:
                warnings.append(fund_resp.error)
                errors.append(fund_resp.error)
            else:
                stock_view.update(
                    {
                        "market_cap": fund_resp.market_cap,
                        "pe_ratio": fund_resp.pe_ratio,
                        "dividend_yield": fund_resp.dividend_yield,
                        "beta": fund_resp.beta,
                    }
                )
                trend_view.append(
                    f"Fundamentals: market cap {fund_resp.market_cap}, P/E {fund_resp.pe_ratio}, beta {fund_resp.beta}."
                )
                evidence_lines.append(trend_view[-1])
                sources.append(f"yfinance: {ticker} fundamentals")
                source_items.append(
                    SourceItem(
                        title=f"{_company_label(primary)} fundamentals",
                        source_type="fundamentals",
                        source=f"yfinance: {ticker} fundamentals",
                        snippet=trend_view[-1],
                    )
                )

        if (plan.requires_news or "search_news" in plan.tool_sequence) and plan.intent != "compare":
            news_query = company_name or ticker or query
            news_resp, trace = _run_tool(
                "search_news",
                {"query": news_query, "max_results": 5},
                search_news,
                news_query,
                max_results=5,
            )
            tool_calls.append(trace)
            if news_resp.error:
                warnings.append(news_resp.error)
                errors.append(news_resp.error)
            for article in news_resp.articles:
                bullet = f"{article.title} ({article.source})"
                news_view.append(bullet)
                evidence_lines.append(bullet)
                sources.append(article.url)
                source_items.append(
                    SourceItem(
                        title=article.title,
                        source_type="news",
                        source=article.source,
                        url=article.url,
                        date=article.published_date,
                        snippet=article.summary,
                    )
                )

        if "get_india_market_ideas" in plan.tool_sequence:
            ideas_resp, trace = _run_tool(
                "get_india_market_ideas",
                {"query": query, "max_results": 5},
                get_india_market_ideas,
                query,
                max_results=5,
            )
            tool_calls.append(trace)
            if ideas_resp.error:
                warnings.append(ideas_resp.error)
                errors.append(ideas_resp.error)
            if ideas_resp.market_snapshot:
                stock_view.update(
                    {
                        "india_universe_size": ideas_resp.market_snapshot.get("universe_size"),
                        "india_screened": ideas_resp.market_snapshot.get("screened"),
                        "india_risk_profile": ideas_resp.market_snapshot.get("risk_profile"),
                        "india_advancers": ideas_resp.market_snapshot.get("market_breadth", {}).get("advancers"),
                        "india_decliners": ideas_resp.market_snapshot.get("market_breadth", {}).get("decliners"),
                        "india_nifty_1m_pct": ideas_resp.market_snapshot.get("index_snapshot_1m_pct", {}).get("NIFTY50"),
                        "india_sensex_1m_pct": ideas_resp.market_snapshot.get("index_snapshot_1m_pct", {}).get("SENSEX"),
                    }
                )
                leaders = ideas_resp.market_snapshot.get("sector_leaders", [])
                if leaders:
                    trend_view.append(f"India sector leadership snapshot: {', '.join(leaders)}.")
                    evidence_lines.append(trend_view[-1])
            if ideas_resp.ideas:
                best = ideas_resp.ideas[0]
                lead_in = "Top India candidate" if plan.intent == "market_ideas" else "Representative market leader"
                trend_view.append(f"{lead_in}: {best.company_name} ({best.ticker}) with action {best.action} and score {best.score:.2f}.")
                evidence_lines.append(trend_view[-1])
                for idea in ideas_resp.ideas:
                    bullet = (
                        f"{idea.company_name} ({idea.ticker}): {idea.action} | "
                        f"1M change {idea.change_pct_1m:+.2f}% | {idea.rationale}"
                    )
                    news_view.append(bullet)
                    evidence_lines.append(bullet)
                    sources.append(f"yfinance: {idea.ticker} 1mo OHLCV")
                    source_items.append(
                        SourceItem(
                            title=f"{idea.company_name} screening snapshot",
                            source_type="market_data",
                            source=f"yfinance: {idea.ticker} 1mo OHLCV",
                            snippet=bullet,
                        )
                    )
                    if idea.action in {"SELL", "AVOID"}:
                        risk_view.append(
                            f"{idea.company_name} flagged as {idea.action.lower()} due to weak momentum profile."
                        )
                warnings.append(ideas_resp.disclaimer)

        if plan.requires_rag:
            retriever = get_finance_retriever()
            rag_query = f"{company_name or ticker or ''} {query}".strip()
            retrieval, trace = _run_tool(
                "rag_retriever",
                {"query": rag_query, "top_k": 4, "ticker": ticker},
                retriever.retrieve,
                rag_query,
                top_k=4,
                ticker=ticker,
            )
            tool_calls.append(trace)
            if retrieval.error:
                warnings.append(retrieval.error)
                errors.append(retrieval.error)
            for item in retrieval.results:
                snippet = " ".join(item.text.split())
                clipped = snippet[:220] + ("..." if len(snippet) > 220 else "")
                evidence_line = f"{item.title}: {clipped}"
                evidence_lines.append(evidence_line)
                risk_view.append(clipped.split(".")[0])
                src = item.metadata.get("source_url") or item.source
                sources.append(src)
                source_items.append(
                    SourceItem(
                        title=item.title,
                        source_type=item.source_type or "retrieval",
                        source=item.source,
                        url=item.metadata.get("source_url"),
                        date=item.document_date,
                        snippet=clipped,
                    )
                )

        if not summary:
            summary, model_recommendation, model_rationale = synthesize_grounded_response(
                query=query,
                subject=_company_label(primary),
                tool_summary="\n".join(trend_view + news_view),
                evidence_lines=evidence_lines,
                model_name=model_name,
            )
            recommendation = model_recommendation
            if not decision_rationale:
                decision_rationale = model_rationale

        successful_traces = [trace for trace in tool_calls if trace.success]
        failed_tools = [trace.tool_name for trace in tool_calls if not trace.success]
        if failed_tools:
            warnings.append(
                f"Partial data: {', '.join(_dedupe(failed_tools))} returned errors. "
                "Analysis based on available sources only."
            )

        sources = _dedupe(sources)
        source_items = _dedupe_source_items([item for item in source_items if item.source])
        relevance = _query_relevance(query, evidence_lines)
        unsupported_claims = 0 if evidence_lines else 1
        source_types = [item.source_type for item in source_items]
        grounding_score, quality_warnings = assess_grounding(
            evidence_lines=evidence_lines,
            sources=sources,
            tool_trace_count=len(successful_traces),
            source_types=source_types,
            query_relevance=relevance,
            unsupported_claims=unsupported_claims,
            expected_claims=max(1, len((summary or "").split("."))),
        )
        warnings.extend(quality_warnings)
        summary = enforce_grounded_wording(
            thesis=summary,
            warnings=quality_warnings,
            threshold=settings.low_grounding_threshold,
            score=grounding_score,
        )

        base_confidence = 0.78 if successful_traces else 0.35
        calibrated_confidence = calibrate_confidence(
            base_confidence=base_confidence,
            grounding_score=grounding_score,
            has_error=False,
        )
        if recommendation not in {"BUY", "SELL", "HOLD"}:
            recommendation = "HOLD"
        if not decision_rationale:
            decision_rationale = (
                f"Recommendation derived from intent={plan.intent}, grounding={grounding_score:.2f}, "
                f"confidence={calibrated_confidence:.2f}."
            )
        if not bull_case and not bear_case:
            bull_case, bear_case = _extract_cases(trend_view, risk_view, news_view)

        citations = _build_citations(source_items, ticker)
        latency_ms = (time.time() - started) * 1000

        answer = AnalystAnswer(
            query=query,
            ticker=ticker,
            company_name=company_name,
            intent=plan.intent,
            summary=summary,
            bull_case=bull_case,
            bear_case=bear_case,
            decision_rationale=decision_rationale,
            comparison_view=comparison_view,
            source_items=source_items,
            chart_data=chart_data,
            stock_view=stock_view,
            news_view=news_view,
            trend_view=trend_view,
            risk_view=_dedupe([item for item in risk_view if item]),
            recommendation=recommendation,
            recommendation_confidence=calibrated_confidence,
            grounding_score=grounding_score,
            tool_calls=tool_calls,
            citations=citations,
            source_count=len(sources),
            latency_ms=latency_ms,
            warnings=_dedupe(warnings + [plan.reasoning]),
            error="; ".join(_dedupe(errors)) if errors else None,
        )
        return answer

    except Exception as exc:
        logger.error("Error in route_query: %s", exc)
        latency_ms = (time.time() - started) * 1000
        answer = AnalystAnswer(
            query=query,
            ticker=ticker,
            company_name=company_name,
            intent=plan.intent,
            summary="Unable to complete the full analysis. Showing partial output only.",
            bull_case=bull_case,
            bear_case=bear_case,
            decision_rationale=decision_rationale,
            comparison_view=comparison_view,
            source_items=source_items,
            chart_data=chart_data,
            stock_view=stock_view,
            news_view=news_view,
            trend_view=trend_view,
            risk_view=risk_view,
            recommendation="HOLD",
            recommendation_confidence=0.0,
            grounding_score=0.0,
            tool_calls=tool_calls,
            citations=_build_citations(_dedupe_source_items(source_items), ticker),
            source_count=len(_dedupe(sources)),
            latency_ms=latency_ms,
            warnings=_dedupe(warnings + [str(exc)]),
            error=str(exc),
        )
        return answer


def extract_ticker(query: str) -> str | None:
    primary_entity = resolve_primary_company(query)
    return primary_entity.ticker if primary_entity else None
