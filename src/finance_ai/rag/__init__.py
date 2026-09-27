from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "load_documents": ("finance_ai.rag.chunking", "load_documents"),
    "split_documents": ("finance_ai.rag.chunking", "split_documents"),
    "SimpleKeywordEmbeddings": ("finance_ai.rag.embeddings", "SimpleKeywordEmbeddings"),
    "SentenceTransformerEmbeddings": ("finance_ai.rag.embeddings", "SentenceTransformerEmbeddings"),
    "get_finance_retriever": ("finance_ai.rag.service", "get_finance_retriever"),
    "FinanceRetriever": ("finance_ai.rag.retriever", "FinanceRetriever"),
    "build_finance_retriever": ("finance_ai.rag.retriever", "build_finance_retriever"),
    "build_vector_store": ("finance_ai.rag.vector_store", "build_vector_store"),
    "load_vector_store": ("finance_ai.rag.vector_store", "load_vector_store"),
    "save_vector_store": ("finance_ai.rag.vector_store", "save_vector_store"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


__all__ = list(_EXPORTS)
