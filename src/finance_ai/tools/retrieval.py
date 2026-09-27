from __future__ import annotations

from finance_ai.rag.service import get_finance_retriever
from finance_ai.schemas.rag import RetrievalResponse


def retrieve_finance_context(
    query: str,
    top_k: int = 4,
    ticker: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> RetrievalResponse:
    retriever = get_finance_retriever()
    return retriever.retrieve(
        query=query,
        top_k=top_k,
        ticker=ticker,
        date_from=date_from,
        date_to=date_to,
    )
