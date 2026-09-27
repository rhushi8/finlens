from pathlib import Path

from langchain_core.embeddings import Embeddings

from finance_ai.config import get_settings
from finance_ai.rag.chunking import load_documents, split_documents
from finance_ai.rag.embeddings import SentenceTransformerEmbeddings
from finance_ai.rag.reranker import FinanceReranker
from finance_ai.rag.vector_store import build_vector_store, load_vector_store, save_vector_store
from finance_ai.schemas.rag import RetrievedChunk, RetrievalResponse


class FinanceRetriever:
    def __init__(
        self,
        vector_store,
        embeddings: Embeddings,
        source_paths: list[str] | None = None,
        reranker: FinanceReranker | None = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.source_paths = source_paths or []
        self.reranker = reranker

    @classmethod
    def from_source_paths(
        cls,
        source_paths: list[str | Path],
        embeddings: Embeddings | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> "FinanceRetriever":
        embeddings = embeddings or SentenceTransformerEmbeddings()
        documents = load_documents(source_paths)
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vector_store = build_vector_store(chunks, embeddings)
        settings = get_settings()
        reranker = FinanceReranker() if settings.enable_rerank else None
        return cls(
            vector_store=vector_store,
            embeddings=embeddings,
            source_paths=[str(path) for path in source_paths],
            reranker=reranker,
        )

    @classmethod
    def from_index(
        cls,
        index_path: str | Path,
        embeddings: Embeddings | None = None,
    ) -> "FinanceRetriever":
        embeddings = embeddings or SentenceTransformerEmbeddings()
        vector_store = load_vector_store(index_path, embeddings)
        settings = get_settings()
        reranker = FinanceReranker() if settings.enable_rerank else None
        return cls(vector_store=vector_store, embeddings=embeddings, reranker=reranker)

    def save(self, index_path: str | Path) -> None:
        save_vector_store(self.vector_store, index_path)

    @staticmethod
    def _l2_to_similarity(distance: float) -> float:
        value = max(0.0, float(distance))
        return 1.0 / (1.0 + value)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        ticker: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> RetrievalResponse:
        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=max(top_k * 3, 8))
            results: list[RetrievedChunk] = []
            seen_signatures: set[str] = set()

            def _passes_filters(metadata: dict) -> bool:
                doc_ticker = str(metadata.get("ticker") or "").upper()
                if ticker and doc_ticker and doc_ticker != ticker.upper():
                    return False

                doc_date = metadata.get("document_date")
                if doc_date and date_from and str(doc_date) < str(date_from):
                    return False
                if doc_date and date_to and str(doc_date) > str(date_to):
                    return False
                return True

            for doc, score in docs_and_scores:
                metadata = dict(doc.metadata)
                if not _passes_filters(metadata):
                    continue

                signature = f"{metadata.get('source')}::{metadata.get('chunk_index')}"
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)

                adjusted_score = self._l2_to_similarity(float(score))
                if ticker and str(metadata.get("ticker") or "").upper() == ticker.upper():
                    adjusted_score *= 1.10
                if metadata.get("source_type") == "filing":
                    adjusted_score *= 1.03

                results.append(
                    RetrievedChunk(
                        text=doc.page_content,
                        source=metadata.get("source", "unknown"),
                        title=metadata.get("title", "unknown"),
                        score=adjusted_score,
                        ticker=metadata.get("ticker"),
                        source_type=metadata.get("source_type"),
                        document_date=metadata.get("document_date"),
                        source_url=metadata.get("source_url"),
                        metadata=metadata,
                    )
                )

            results = sorted(results, key=lambda item: item.score, reverse=True)
            if self.reranker:
                results = self.reranker.rerank(query=query, chunks=results, top_k=top_k)
            else:
                results = results[:top_k]

            return RetrievalResponse(query=query, results=results)
        except Exception as exc:
            return RetrievalResponse(query=query, results=[], error=str(exc))


def build_finance_retriever(source_paths: list[str | Path], embeddings: Embeddings | None = None) -> FinanceRetriever:
    return FinanceRetriever.from_source_paths(source_paths=source_paths, embeddings=embeddings)