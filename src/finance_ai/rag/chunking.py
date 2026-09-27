import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finance_ai.utils.company_resolution import COMPANY_ALIASES, TICKER_TO_COMPANY


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
MIN_CHUNK_CHARS = 600


def _extract_ticker_from_name(file_path: Path) -> str | None:
    stem_upper = file_path.stem.upper()
    raw_candidates = re.findall(
        r"([A-Z][A-Z0-9]{0,11})(?:[\._-](NS|BO))?(?=[^A-Z0-9]|$)",
        stem_upper,
    )
    for base, suffix in raw_candidates:
        candidate = f"{base}.{suffix}" if suffix else base
        if candidate in TICKER_TO_COMPANY:
            return candidate

    normalized_stem = re.sub(r"[^a-z0-9]+", " ", file_path.stem.lower()).strip()
    if normalized_stem:
        for alias, ticker, _company_name in COMPANY_ALIASES:
            normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
            if normalized_alias and re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_stem):
                return ticker
    return None


def _extract_doc_date_from_name(file_path: Path) -> str | None:
    match = re.search(r"(20\d{2}[-_]?\d{2}[-_]?\d{2})", file_path.stem)
    if not match:
        return None
    value = match.group(1).replace("_", "-")
    if len(value) == 8:
        return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"
    return value


def _guess_source_type(file_path: Path) -> str:
    lower_path = str(file_path).lower()
    if "filing" in lower_path or "10-k" in lower_path or "10q" in lower_path:
        return "filing"
    if "news" in lower_path:
        return "news"
    if "note" in lower_path or "research" in lower_path:
        return "research_note"
    return "document"


def _extract_source_url(content: str) -> str | None:
    match = re.search(r"https?://\S+", content)
    return match.group(0).rstrip(").,") if match else None


def _load_text_file(file_path: Path) -> tuple[str, dict[str, object]]:
    content = file_path.read_text(encoding="utf-8")
    return content, {"source_url": _extract_source_url(content)}


def _load_pdf_file(file_path: Path) -> tuple[str, dict[str, object]]:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())

    content = "\n\n".join(pages)
    return content, {"page_count": len(reader.pages)}


def load_documents(source_dirs: list[str | Path]) -> list[Document]:
    documents: list[Document] = []

    for source_dir in source_dirs:
        base_path = Path(source_dir)
        if not base_path.exists():
            continue

        for file_path in base_path.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            if file_path.suffix.lower() == ".pdf":
                content, extra_metadata = _load_pdf_file(file_path)
            else:
                content, extra_metadata = _load_text_file(file_path)

            if not content.strip():
                continue

            metadata = {
                "source": str(file_path),
                "title": file_path.stem,
                "extension": file_path.suffix.lower(),
                "ticker": _extract_ticker_from_name(file_path),
                "source_type": _guess_source_type(file_path),
                "document_date": _extract_doc_date_from_name(file_path),
            }
            metadata.update(extra_metadata)
            documents.append(Document(page_content=content, metadata=metadata))

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    small_docs = [doc for doc in documents if len((doc.page_content or "").strip()) < MIN_CHUNK_CHARS]
    large_docs = [doc for doc in documents if len((doc.page_content or "").strip()) >= MIN_CHUNK_CHARS]

    chunks: list[Document] = []
    for document in small_docs:
        metadata = dict(document.metadata)
        metadata.update(
            {
                "chunk_index": 0,
                "chunk_size": chunk_size,
                "chunk_overlap": 0,
                "chunking_strategy": "atomic_small_doc",
            }
        )
        chunks.append(Document(page_content=document.page_content, metadata=metadata))

    if not large_docs:
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for document in large_docs:
        split_docs = splitter.split_documents([document])
        for index, chunk in enumerate(split_docs):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunking_strategy": "split",
                }
            )
            chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

    return chunks