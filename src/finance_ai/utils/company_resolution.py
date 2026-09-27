from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedCompany:
    ticker: str
    company_name: str
    matched_text: str
    match_type: str
    sort_key: int


COMPANY_ALIASES: list[tuple[str, str, str]] = [
    ("bank of america", "BAC", "Bank of America"),
    ("bofa", "BAC", "Bank of America"),
    ("apple", "AAPL", "Apple"),
    ("apple inc", "AAPL", "Apple"),
    ("apple computer", "AAPL", "Apple"),
    ("nvidia", "NVDA", "NVIDIA"),
    ("tesla", "TSLA", "Tesla"),
    ("microsoft", "MSFT", "Microsoft"),
    ("microsoft corporation", "MSFT", "Microsoft"),
    ("microsoft corp", "MSFT", "Microsoft"),
    ("alphabet", "GOOGL", "Alphabet"),
    ("alphabet inc", "GOOGL", "Alphabet"),
    ("google", "GOOGL", "Alphabet"),
    ("amazon", "AMZN", "Amazon"),
    ("amazon.com", "AMZN", "Amazon"),
    ("meta", "META", "Meta"),
    ("meta platforms", "META", "Meta"),
    ("facebook", "META", "Meta"),
    ("netflix", "NFLX", "Netflix"),
    ("advanced micro devices", "AMD", "Advanced Micro Devices"),
    ("amd", "AMD", "Advanced Micro Devices"),
    ("intel", "INTC", "Intel"),
    ("qualcomm", "QCOM", "Qualcomm"),
    ("cisco", "CSCO", "Cisco"),
    ("oracle", "ORCL", "Oracle"),
    ("uber", "UBER", "Uber"),
    ("shopify", "SHOP", "Shopify"),
    ("walmart", "WMT", "Walmart"),
    ("disney", "DIS", "Disney"),
    ("adobe", "ADBE", "Adobe"),
    ("salesforce", "CRM", "Salesforce"),
    ("palantir", "PLTR", "Palantir"),
    ("broadcom", "AVGO", "Broadcom"),
    ("coca cola", "KO", "Coca-Cola"),
    ("coca-cola", "KO", "Coca-Cola"),
    ("pepsi", "PEP", "PepsiCo"),
    ("jpmorgan", "JPM", "JPMorgan Chase"),
    ("jp morgan", "JPM", "JPMorgan Chase"),
    ("nifty", "^NSEI", "NIFTY 50"),
    ("nifty 50", "^NSEI", "NIFTY 50"),
    ("nifty50", "^NSEI", "NIFTY 50"),
    ("sensex", "^BSESN", "SENSEX"),
    ("bank nifty", "^NSEBANK", "NIFTY BANK"),
    ("banknifty", "^NSEBANK", "NIFTY BANK"),
    ("reliance", "RELIANCE.NS", "Reliance Industries"),
    ("tcs", "TCS.NS", "Tata Consultancy Services"),
    ("infosys", "INFY.NS", "Infosys"),
    ("hdfc bank", "HDFCBANK.NS", "HDFC Bank"),
    ("icici bank", "ICICIBANK.NS", "ICICI Bank"),
    ("sbi", "SBIN.NS", "State Bank of India"),
    ("state bank of india", "SBIN.NS", "State Bank of India"),
    ("lt", "LT.NS", "Larsen & Toubro"),
    ("larsen and toubro", "LT.NS", "Larsen & Toubro"),
    ("bharti airtel", "BHARTIARTL.NS", "Bharti Airtel"),
    ("itc", "ITC.NS", "ITC"),
    ("tata motors", "TATAMOTORS.NS", "Tata Motors"),
    ("adani enterprises", "ADANIENT.NS", "Adani Enterprises"),
    ("bajaj finance", "BAJFINANCE.NS", "Bajaj Finance"),
    ("asian paints", "ASIANPAINT.NS", "Asian Paints"),
    ("maruti", "MARUTI.NS", "Maruti Suzuki"),
]

TICKER_TO_COMPANY: dict[str, str] = {ticker: company_name for _, ticker, company_name in COMPANY_ALIASES}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_ticker_token(token: str) -> bool:
    if token in {"I", "A"}:
        return False
    if token.endswith(".NS") or token.endswith(".BO"):
        base = token.split(".")[0]
        return base.isalnum() and 1 <= len(base) <= 12
    return token.isalpha() and token.isupper() and 1 <= len(token) <= 12


def resolve_company_entities(query: str) -> list[ResolvedCompany]:
    normalized_query = _normalize(query)
    matches: list[ResolvedCompany] = []

    for alias, ticker, company_name in COMPANY_ALIASES:
        normalized_alias = _normalize(alias)
        if not normalized_alias:
            continue

        pattern = rf"\b{re.escape(normalized_alias)}\b"
        match = re.search(pattern, normalized_query)
        if match:
            matches.append(
                ResolvedCompany(
                    ticker=ticker,
                    company_name=company_name,
                    matched_text=alias,
                    match_type="alias",
                    sort_key=match.start(),
                )
            )

    for token_match in re.finditer(r"\b[A-Za-z][A-Za-z0-9\.]{0,14}\b", query):
        token = token_match.group(0)
        if not _is_ticker_token(token):
            continue

        ticker = token.upper()
        matches.append(
            ResolvedCompany(
                ticker=ticker,
                company_name=TICKER_TO_COMPANY.get(ticker, ticker),
                matched_text=token,
                match_type="ticker",
                sort_key=token_match.start(),
            )
        )

    deduped: list[ResolvedCompany] = []
    seen_tickers: set[str] = set()
    for entity in sorted(matches, key=lambda item: item.sort_key):
        if entity.ticker in seen_tickers:
            continue
        seen_tickers.add(entity.ticker)
        deduped.append(entity)

    return deduped


def resolve_primary_company(query: str) -> ResolvedCompany | None:
    entities = resolve_company_entities(query)
    return entities[0] if entities else None