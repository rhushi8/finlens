import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_vector_store(documents: list[Document], embeddings: Embeddings) -> FAISS:
    if not documents:
        raise ValueError("No documents provided to build vector store")
    return FAISS.from_documents(documents, embeddings)


def save_vector_store(vector_store: FAISS, target_dir: str | Path) -> None:
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))


def load_vector_store(target_dir: str | Path, embeddings: Embeddings) -> FAISS:
    path = Path(target_dir)
    allow_dangerous = os.getenv("FINANCE_AI_TRUST_VECTOR_INDEX", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=allow_dangerous,
    )