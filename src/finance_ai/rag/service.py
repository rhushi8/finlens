from __future__ import annotations

import logging
import hashlib
import re
from functools import lru_cache
from pathlib import Path

from finance_ai.config import ROOT_DIR, get_settings
from finance_ai.rag.embeddings import SentenceTransformerEmbeddings, SimpleKeywordEmbeddings
from finance_ai.rag.retriever import FinanceRetriever

logger = logging.getLogger(__name__)

DOC_SOURCE_DIRS = [
    ROOT_DIR / "docs" / "example_notes",
    ROOT_DIR / "docs" / "example_filings",
    ROOT_DIR / "docs" / "example_india",
]
INDEX_SCHEMA_VERSION = "v2"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "index"


def _embedding_mode() -> str:
    return get_settings().embedding_mode.lower().strip()


def _build_embeddings():
    mode = _embedding_mode()
    if mode in {"simple", "keyword", "keywords"}:
        return SimpleKeywordEmbeddings()

    try:
        return SentenceTransformerEmbeddings()
    except Exception as exc:
        logger.warning("Falling back to keyword embeddings: %s", exc)
        return SimpleKeywordEmbeddings()


def _vectorstore_root() -> Path:
    settings = get_settings()
    configured_root = Path(settings.vectorstore_root)
    if configured_root.is_absolute():
        return configured_root
    return ROOT_DIR / configured_root


def _index_path_for_embeddings(embeddings) -> Path:
    if isinstance(embeddings, SentenceTransformerEmbeddings):
        suffix = _slugify(embeddings.model_name)
    else:
        suffix = "simple_keyword"
    return _vectorstore_root() / f"faiss_{suffix}_{INDEX_SCHEMA_VERSION}"


def _source_files(source_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for file_path in source_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(file_path)
    return sorted(files)


def _corpus_fingerprint(source_dirs: list[Path]) -> str:
    digest = hashlib.sha256()
    files = _source_files(source_dirs)
    if not files:
        digest.update(b"empty-corpus")
    for file_path in files:
        stat = file_path.stat()
        digest.update(str(file_path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()[:12]


def _index_path_for_corpus(embeddings, source_dirs: list[Path]) -> Path:
    base = _index_path_for_embeddings(embeddings)
    fingerprint = _corpus_fingerprint(source_dirs)
    return base / f"auto_{fingerprint}"


def _build_or_load_retriever(embeddings) -> FinanceRetriever:
    source_dirs = [Path(path) for path in DOC_SOURCE_DIRS]
    index_path = _index_path_for_corpus(embeddings, source_dirs)
    if index_path.exists() and any(index_path.iterdir()):
        try:
            return FinanceRetriever.from_index(index_path, embeddings=embeddings)
        except Exception as exc:
            logger.warning("Failed to load existing RAG index at %s: %s", index_path, exc)

    retriever = FinanceRetriever.from_source_paths(source_dirs, embeddings=embeddings)
    try:
        retriever.save(index_path)
    except Exception as exc:
        logger.warning("Failed to save RAG index at %s: %s", index_path, exc)
    return retriever


@lru_cache(maxsize=1)
def get_finance_retriever() -> FinanceRetriever:
    attempted_errors: list[Exception] = []
    embeddings_candidates = [_build_embeddings()]
    if not isinstance(embeddings_candidates[0], SimpleKeywordEmbeddings):
        embeddings_candidates.append(SimpleKeywordEmbeddings())

    for embeddings in embeddings_candidates:
        try:
            return _build_or_load_retriever(embeddings)
        except Exception as exc:
            attempted_errors.append(exc)
            logger.warning("Retriever build failed with %s: %s", type(embeddings).__name__, exc)

    raise RuntimeError(f"Unable to initialize finance retriever: {attempted_errors}")