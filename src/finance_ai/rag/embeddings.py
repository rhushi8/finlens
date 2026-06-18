"""Embedding models for finance RAG."""

from __future__ import annotations

import hashlib
import os
from collections import Counter

from langchain_core.embeddings import Embeddings


class SimpleKeywordEmbeddings(Embeddings):
    """Lightweight deterministic embeddings for tests and fallback use."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = [token.lower() for token in text.split()]
        counts = Counter(tokens)
        for token, count in counts.items():
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[index] += float(count)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbeddings(Embeddings):
    """Sentence-transformers based embeddings for semantic retrieval."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or os.getenv("FINANCE_AI_EMBEDDING_DEVICE", "cpu")
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()