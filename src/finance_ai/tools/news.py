"""News retrieval tool using free GDELT DOC API."""

from __future__ import annotations

import logging
from datetime import datetime

import requests
import yfinance as yf

from finance_ai.schemas.tools import NewsArticle, NewsSearchResponse
from finance_ai.utils.cache import cached
from finance_ai.utils.company_resolution import resolve_primary_company

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def _parse_seendate(seendate: str | None) -> datetime | None:
    if not seendate:
        return None

    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(seendate, fmt)
        except ValueError:
            continue
    return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _parse_publish_epoch(value: int | float | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value))
    except (TypeError, ValueError, OSError):
        return None


def _fallback_yfinance_news(query: str, max_results: int) -> NewsSearchResponse:
    """Fallback news provider when GDELT is unavailable or rate-limited."""

    entity = resolve_primary_company(query)
    if not entity:
        return NewsSearchResponse(
            query=query,
            articles=[],
            retrieved_at=datetime.now(),
        )

    try:
        ticker_news = yf.Ticker(entity.ticker).news or []
    except Exception as exc:
        logger.warning("yfinance news fallback failed for '%s': %s", entity.ticker, exc)
        return NewsSearchResponse(
            query=query,
            articles=[],
            retrieved_at=datetime.now(),
        )

    articles: list[NewsArticle] = []
    for item in ticker_news[: max(1, min(max_results, 10))]:
        # yfinance >=0.2.40 nests fields under "content"; older versions are flat.
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = content.get("title")
        url = (
            content.get("link")
            or content.get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or (content.get("canonicalUrl") or {}).get("url")
        )
        if not title or not url:
            continue
        publisher = (
            content.get("publisher")
            or (content.get("provider") or {}).get("displayName")
            or "yfinance"
        )
        published = _parse_publish_epoch(item.get("providerPublishTime")) or _parse_iso_datetime(
            content.get("pubDate") or content.get("displayTime")
        )
        articles.append(
            NewsArticle(
                source=f"{publisher} (yfinance)",
                title=title,
                url=url,
                published_date=published,
                summary=content.get("summary"),
            )
        )

    if articles:
        return NewsSearchResponse(query=query, articles=articles, retrieved_at=datetime.now())

    return NewsSearchResponse(query=query, articles=[], retrieved_at=datetime.now())


@cached(ttl_seconds=600)
def search_news(query: str, max_results: int = 5) -> NewsSearchResponse:
    """Search recent finance news with GDELT DOC API.

    Args:
        query: Search query (company name, ticker, topic)
        max_results: Max articles to return

    Returns:
        NewsSearchResponse with parsed article metadata
    """
    try:
        query = query.strip()
        if not query:
            return NewsSearchResponse(
                query=query,
                articles=[],
                retrieved_at=datetime.now(),
                error="Empty query",
            )

        if max_results <= 0:
            return NewsSearchResponse(
                query=query,
                articles=[],
                retrieved_at=datetime.now(),
                error="max_results must be positive",
            )

        params = {
            "query": f"{query} language:english",
            "mode": "artlist",
            "maxrecords": max(1, min(max_results, 25)),
            "sort": "datedesc",
            "format": "json",
        }

        response = requests.get(GDELT_DOC_API, params=params, timeout=20)
        if response.status_code == 429:
            logger.info("GDELT rate-limited query '%s'; switching to yfinance fallback.", query)
            return _fallback_yfinance_news(query, max_results)
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError:
            logger.info("GDELT non-JSON response for query '%s'; switching to yfinance fallback.", query)
            return _fallback_yfinance_news(query, max_results)

        raw_articles = payload.get("articles", [])
        parsed_articles: list[NewsArticle] = []

        for item in raw_articles:
            url = item.get("url")
            title = item.get("title")
            if not url or not title:
                continue

            domain = item.get("domain") or "gdelt"
            source_country = item.get("sourcecountry")
            source = f"{domain} ({source_country})" if source_country else domain

            summary_parts = []
            if item.get("seendate"):
                summary_parts.append(f"Seen: {item['seendate']}")
            if item.get("tone") is not None:
                summary_parts.append(f"Tone: {item['tone']}")

            parsed_articles.append(
                NewsArticle(
                    source=source,
                    title=title,
                    url=url,
                    published_date=_parse_seendate(item.get("seendate")),
                    summary=" | ".join(summary_parts) if summary_parts else None,
                )
            )

        return NewsSearchResponse(
            query=query,
            articles=parsed_articles,
            retrieved_at=datetime.now(),
        )

    except Exception as exc:
        logger.warning("GDELT news lookup failed for query '%s': %s", query, exc)
        return _fallback_yfinance_news(query, max_results)
