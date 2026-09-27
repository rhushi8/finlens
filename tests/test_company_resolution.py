from finance_ai.agents import extract_ticker
from finance_ai.utils.company_resolution import (
    resolve_company_entities,
    resolve_primary_company,
)


def test_resolve_primary_company_from_name() -> None:
    entity = resolve_primary_company("What is the price of Apple?")
    assert entity is not None
    assert entity.ticker == "AAPL"
    assert entity.company_name == "Apple"


def test_resolve_multiple_companies_for_compare() -> None:
    entities = resolve_company_entities("Compare Apple and Microsoft")
    tickers = [entity.ticker for entity in entities]
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_resolve_formal_company_names() -> None:
    entities = resolve_company_entities("Compare Apple Inc and Microsoft Corporation")
    tickers = [entity.ticker for entity in entities]
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_extract_ticker_uses_company_aliases() -> None:
    assert extract_ticker("What are the risks in Nvidia right now?") == "NVDA"


def test_resolve_indian_company_alias() -> None:
    entity = resolve_primary_company("Should I buy Reliance now?")
    assert entity is not None
    assert entity.ticker == "RELIANCE.NS"


def test_extract_nse_suffix_ticker() -> None:
    assert extract_ticker("What is the trend for TCS.NS today?") == "TCS.NS"