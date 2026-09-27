from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from finance_ai.schemas.tools import ToolTrace


class Citation(BaseModel):
    title: str
    source: str
    url: Optional[str] = None
    ticker: Optional[str] = None
    source_type: Optional[str] = None
    published_at: Optional[datetime] = None
    snippet: Optional[str] = None


class QueryPlan(BaseModel):
    intent: Literal["price", "fundamentals", "news", "rag", "compare", "market_ideas", "market_general", "unknown"]
    is_comparison: bool = False
    requires_rag: bool = False
    requires_news: bool = False
    response_style: Literal["short", "detailed"] = "short"
    confidence_low: bool = False
    tool_sequence: list[str] = Field(default_factory=list)
    reasoning: str = ""
    execution_steps: list[dict[str, str]] = Field(default_factory=list)
    response_sections: list[str] = Field(default_factory=list)
    planning_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    tool_choice_rationale: str = ""
    fallback_tools: list[str] = Field(default_factory=list)


class ComparisonLeg(BaseModel):
    ticker: str
    company_name: str
    price: Optional[float] = None
    change_pct_1m: Optional[float] = None
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    news_highlights: list[str] = Field(default_factory=list)
    bull_points: list[str] = Field(default_factory=list)
    bear_points: list[str] = Field(default_factory=list)


class ComparisonView(BaseModel):
    left: ComparisonLeg
    right: ComparisonLeg
    winner: str = "HOLD"
    recommendation: str = "HOLD"
    winner_reason: str = ""
    key_differences: list[str] = Field(default_factory=list)


class SourceItem(BaseModel):
    title: str
    source_type: str = "market_or_doc"
    source: str
    url: Optional[str] = None
    date: Optional[datetime] = None
    snippet: Optional[str] = None


class AnalystAnswer(BaseModel):
    query: str
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    intent: str = "unknown"
    summary: str = Field(default="", description="Main insight or answer")
    bull_case: list[str] = Field(default_factory=list)
    bear_case: list[str] = Field(default_factory=list)
    decision_rationale: str = ""
    comparison_view: Optional[ComparisonView] = None
    source_items: list[SourceItem] = Field(default_factory=list)
    chart_data: dict[str, Any] = Field(default_factory=dict)
    stock_view: dict[str, Any] = Field(default_factory=dict)
    news_view: list[str] = Field(default_factory=list)
    trend_view: list[str] = Field(default_factory=list)
    risk_view: list[str] = Field(default_factory=list)
    recommendation: str = "HOLD"
    recommendation_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    source_count: int = 0
    latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    # Kept for older UI/tests.
    thesis: str = Field(default="", description="Legacy alias of summary")
    key_metrics: dict = Field(default_factory=dict, description="Legacy metrics field")
    risks: list[str] = Field(default_factory=list, description="Legacy risk list")
    sources: list[str] = Field(default_factory=list, description="Legacy source list")
    tool_trace: list[ToolTrace] = Field(default_factory=list, description="Legacy tool trace")
    generated_at: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    error: Optional[str] = None

    def sync_legacy_fields(self) -> "AnalystAnswer":
        if not self.thesis:
            self.thesis = self.summary
        if not self.risks:
            self.risks = list(self.risk_view)
        if not self.sources:
            self.sources = [citation.source for citation in self.citations]
        if not self.tool_trace:
            self.tool_trace = list(self.tool_calls)
        if not self.key_metrics:
            self.key_metrics = {
                **self.stock_view,
                "grounding_score": self.grounding_score,
            }
        return self
