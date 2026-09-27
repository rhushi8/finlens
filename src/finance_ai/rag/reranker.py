from __future__ import annotations

import logging

from finance_ai.config import get_settings
from finance_ai.schemas.rag import RetrievedChunk

logger = logging.getLogger(__name__)


class FinanceReranker:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.rerank_model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return chunks

        try:
            model = self._load()
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = model.predict(pairs)
            reranked = list(chunks)
            for idx, chunk in enumerate(reranked):
                chunk.score = float(scores[idx])
            reranked.sort(key=lambda item: item.score, reverse=True)
            return reranked[:top_k]
        except Exception as exc:
            logger.warning("Reranker unavailable, using similarity ordering: %s", exc)
            return chunks[:top_k]
