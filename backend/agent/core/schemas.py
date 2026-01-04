"""
Pydantic schemas for the agentic layer.

Defines request/response models that match the frontend contract.
"""

from typing import Any
from pydantic import BaseModel, Field


# =============================================================================
# Source Models
# =============================================================================


class Source(BaseModel):
    """A source reference for citations."""

    type: str  # "company", "doc", "activity", "opportunity", "history"
    id: str
    label: str


# =============================================================================
# Step Models
# =============================================================================


class Step(BaseModel):
    """A processing step for UI progress display."""

    id: str
    label: str
    status: str = "done"  # "done", "error", "skipped"


# =============================================================================
# Raw Data Models (flexible for UI)
# =============================================================================


class RawData(BaseModel):
    """Raw data payload for UI display."""

    companies: list[dict[str, Any]] = Field(default_factory=list)
    activities: list[dict[str, Any]] = Field(default_factory=list)
    opportunities: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    renewals: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_summary: dict[str, Any] | None = None


# =============================================================================
# Meta Info
# =============================================================================


class MetaInfo(BaseModel):
    """Metadata about the response."""

    mode_used: str  # "docs", "data", "data+docs"
    latency_ms: int
    company_id: str | None = None
    days: int | None = None
    # Per-node latency breakdown (for performance monitoring)
    router_latency_ms: int | None = None
    fetch_latency_ms: int | None = None
    answer_latency_ms: int | None = None
    followup_latency_ms: int | None = None


# =============================================================================
# Request/Response
# =============================================================================


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    question: str = Field(..., min_length=1)
    mode: str | None = "auto"  # "auto", "docs", "data", "data+docs"
    session_id: str | None = None
    user_id: str | None = None
    company_id: str | None = None


class ChatResponse(BaseModel):
    """Response to the frontend."""

    answer: str
    sources: list[Source]
    steps: list[Step]
    raw_data: RawData
    meta: MetaInfo
    follow_up_suggestions: list[str] = Field(default_factory=list)  # Suggested next questions


# =============================================================================
# Router Models
# =============================================================================


class RouterResult(BaseModel):
    """Result from the router's analysis."""

    mode_used: str  # "docs", "data", "data+docs"
    company_id: str | None = None
    company_name_query: str | None = None  # If we need to resolve a name
    days: int = 90
    intent: str = "general"  # "company_status", "renewals", "pipeline", "docs", "general"
    # Role-based owner for filtering (set from starter detection)
    owner: str | None = None  # e.g., "jsmith" for sales rep, "amartin" for CSM
    # LLM router additions (merged routing + understanding)
    query_expansion: str | None = None  # LLM-expanded version of query
    llm_confidence: float | None = None  # Routing confidence (0-1)
    key_entities: list[str] = Field(default_factory=list)  # Extracted entities
    action_type: str | None = None  # "retrieve", "summarize", "compare", "analyze"


# =============================================================================
# Tool Results
# =============================================================================


class ToolResult(BaseModel):
    """Result from a tool function."""

    data: dict[str, Any]
    sources: list[Source]
    error: str | None = None


__all__ = [
    "Source",
    "Step",
    "RawData",
    "MetaInfo",
    "ChatRequest",
    "ChatResponse",
    "RouterResult",
    "ToolResult",
]
