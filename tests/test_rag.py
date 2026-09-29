from pathlib import Path

import pytest
from langchain_core.documents import Document

from finance_ai.rag.chunking import load_documents, split_documents
from finance_ai.rag.embeddings import SimpleKeywordEmbeddings
from finance_ai.rag.retriever import FinanceRetriever, build_finance_retriever
from finance_ai.rag.service import _corpus_fingerprint
from finance_ai.rag.vector_store import build_vector_store, load_vector_store, save_vector_store


@pytest.mark.unit
def test_load_documents_reads_markdown_files() -> None:
    docs = load_documents([
        Path("docs/example_notes"),
        Path("docs/example_filings"),
    ])
    assert len(docs) >= 2
    assert all("source" in doc.metadata for doc in docs)


@pytest.mark.unit
def test_split_documents_creates_chunks() -> None:
    docs = load_documents([
        Path("docs/example_notes"),
    ])
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


@pytest.mark.unit
def test_build_vector_store_and_retrieve_risk_content() -> None:
    embeddings = SimpleKeywordEmbeddings(dimension=64)
    retriever = build_finance_retriever(
        [Path("docs/example_notes"), Path("docs/example_filings")],
        embeddings=embeddings,
    )

    response = retriever.retrieve("What are the risks for Apple?", top_k=3, ticker="AAPL")
    assert response.error is None
    assert len(response.results) > 0
    assert all(result.ticker == "AAPL" for result in response.results)
    combined_text = " ".join(result.text.lower() for result in response.results)
    assert "apple" in combined_text or "risk" in combined_text


@pytest.mark.unit
def test_build_vector_store_requires_documents() -> None:
    embeddings = SimpleKeywordEmbeddings(dimension=32)
    with pytest.raises(ValueError):
        build_vector_store([], embeddings)


def test_save_and_load_vector_store_roundtrip(tmp_path: Path, monkeypatch) -> None:
    docs = load_documents([Path("docs/example_notes")])
    chunks = split_documents(docs, chunk_size=120, chunk_overlap=20)
    embeddings = SimpleKeywordEmbeddings(dimension=64)
    vector_store = build_vector_store(chunks, embeddings)

    index_dir = tmp_path / "faiss_roundtrip"
    save_vector_store(vector_store, index_dir)
    monkeypatch.setenv("FINANCE_AI_TRUST_VECTOR_INDEX", "true")
    loaded_vector_store = load_vector_store(index_dir, embeddings)

    results = loaded_vector_store.similarity_search("Apple supply chain risk", k=2)
    assert len(results) > 0


def test_load_documents_extracts_india_ticker_metadata() -> None:
    docs = load_documents([Path("docs/example_india")])
    assert len(docs) >= 1
    assert any(doc.metadata.get("ticker") == "RELIANCE.NS" for doc in docs)


def test_load_documents_extracts_company_alias_metadata() -> None:
    docs = load_documents([Path("docs/example_filings"), Path("docs/example_notes")])
    tickers = {doc.metadata.get("ticker") for doc in docs}
    assert "AAPL" in tickers
    assert "TSLA" in tickers
    assert "NVDA" in tickers


@pytest.mark.unit
def test_retriever_uses_similarity_semantics_with_descending_rank() -> None:
    class _FakeDoc:
        def __init__(self, metadata: dict[str, str], text: str):
            self.metadata = metadata
            self.page_content = text

    class _FakeStore:
        def similarity_search_with_score(self, query: str, k: int = 8):
            return [
                (_FakeDoc({"source": "a", "chunk_index": 0, "ticker": "AAPL", "title": "Apple", "source_type": "filing"}, "apple filing"), 0.40),
                (_FakeDoc({"source": "b", "chunk_index": 0, "ticker": "", "title": "Macro Note", "source_type": "document"}, "macro note"), 1.20),
            ]

    retriever = FinanceRetriever(vector_store=_FakeStore(), embeddings=SimpleKeywordEmbeddings(dimension=16))
    response = retriever.retrieve("apple outlook", top_k=2, ticker="AAPL")

    assert response.error is None
    assert len(response.results) == 2
    assert response.results[0].ticker == "AAPL"
    assert response.results[0].score > response.results[1].score


@pytest.mark.unit
def test_split_documents_keeps_small_docs_atomic() -> None:
    small = Document(page_content="x" * 200, metadata={"source": "small.md", "title": "small"})
    large = Document(page_content="y" * 1500, metadata={"source": "large.md", "title": "large"})

    chunks = split_documents([small, large], chunk_size=800, chunk_overlap=150)
    small_chunks = [chunk for chunk in chunks if chunk.metadata.get("source") == "small.md"]

    assert len(small_chunks) == 1
    assert small_chunks[0].metadata.get("chunking_strategy") == "atomic_small_doc"
    assert small_chunks[0].metadata.get("chunk_overlap") == 0


@pytest.mark.unit
def test_corpus_fingerprint_changes_on_file_update(tmp_path: Path) -> None:
    file_path = tmp_path / "note_2026-04-09.md"
    file_path.write_text("hello", encoding="utf-8")

    fp1 = _corpus_fingerprint([tmp_path])
    file_path.write_text("hello world", encoding="utf-8")
    fp2 = _corpus_fingerprint([tmp_path])

    assert fp1 != fp2