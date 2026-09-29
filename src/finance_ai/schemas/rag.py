from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


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
