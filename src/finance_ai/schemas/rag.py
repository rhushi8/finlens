from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    title: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 4
    ticker: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class RetrievedChunk(BaseModel):
    text: str
    source: str
    title: str
    score: float
    ticker: Optional[str] = None
    source_type: Optional[str] = None
    document_date: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievedChunk] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
