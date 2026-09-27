from __future__ import annotations

from collections import Counter
from datetime import datetime

from finance_ai.data import INDIA_INDEXES, INDIA_UNIVERSE
from finance_ai.schemas.tools import IndiaMarketIdeasResponse, MarketIdea
from finance_ai.tools.stock import get_stock_price
from finance_ai.utils.cache import cached


def _risk_profile_from_query(query: str) -> str:
    lower = query.lower()
    if any(token in lower for token in {"safe", "conservative", "low risk"}):
        return "conservative"
    if any(token in lower for token in {"aggressive", "high risk", "momentum"}):
        return "aggressive"
    return "moderate"


def _idea_action(change_pct: float, risk_profile: str) -> str:
    if risk_profile == "conservative":
        if change_pct > 2.5:
            return "HOLD"
        if change_pct < -3.0:
            return "AVOID"
        return "WATCH"

    if risk_profile == "aggressive":
        if change_pct > 4.0:
            return "BUY"
        if change_pct < -4.5:
            return "SELL"
        return "WATCH"

    if change_pct > 3.0:
        return "BUY"
    if change_pct < -4.0:
        return "SELL"
    return "HOLD"


@cached(ttl_seconds=900)
def get_india_market_ideas(query: str, max_results: int = 5) -> IndiaMarketIdeasResponse:
    """Heuristic and educational, not personalized advice."""

    risk_profile = _risk_profile_from_query(query)
    candidates: list[MarketIdea] = []
    warning_messages: list[str] = []
    advancers = 0
    decliners = 0
    flat = 0

    for item in INDIA_UNIVERSE:
        ticker = item["ticker"]
        company_name = item["company_name"]
        sector = item["sector"]
        indexes = item["indexes"]

        response = get_stock_price(ticker, period="1mo")
        if response.error:
            warning_messages.append(f"{ticker}: {response.error}")
            continue

        change_pct = float(response.change_pct)
        if change_pct > 0.5:
            advancers += 1
        elif change_pct < -0.5:
            decliners += 1
        else:
            flat += 1

        action = _idea_action(change_pct, risk_profile)
        score = round((change_pct / 10.0) + (0.2 if action == "BUY" else 0.0), 3)
        rationale = (
            f"1-month momentum {change_pct:+.2f}%. "
            f"Sector: {sector}. Current price {response.current_price:.2f} and action bucket '{action}'."
        )
        candidates.append(
            MarketIdea(
                ticker=ticker,
                company_name=company_name,
                action=action,
                score=score,
                rationale=rationale,
                change_pct_1m=change_pct,
                sector=sector,
                index_membership=list(indexes),
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    top_candidates = candidates[: max(1, min(max_results, 10))]

    sector_counts = Counter(idea.sector or "Unknown" for idea in top_candidates)
    sector_leaders = [f"{sector}: {count}" for sector, count in sector_counts.most_common(3)]

    index_snapshot: dict[str, float | str] = {}
    for index_name, ticker in INDIA_INDEXES.items():
        idx_resp = get_stock_price(ticker, period="1mo")
        if idx_resp.error:
            warning_messages.append(f"{index_name}: {idx_resp.error}")
        else:
            index_snapshot[index_name] = idx_resp.change_pct

    snapshot = {
        "universe_size": len(INDIA_UNIVERSE),
        "screened": len(candidates),
        "risk_profile": risk_profile,
        "market_breadth": {
            "advancers": advancers,
            "decliners": decliners,
            "flat": flat,
        },
        "index_snapshot_1m_pct": index_snapshot,
        "sector_leaders": sector_leaders,
        "as_of": datetime.now().isoformat(),
        "warnings": warning_messages[:5],
    }

    return IndiaMarketIdeasResponse(
        query=query,
        market_snapshot=snapshot,
        ideas=top_candidates,
        retrieved_at=datetime.now(),
        error="No market ideas could be generated" if not top_candidates else None,
    )
