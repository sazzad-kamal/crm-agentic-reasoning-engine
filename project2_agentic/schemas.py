"""
Pydantic schemas for the agentic layer.

Defines request/response models that match the frontend contract.
"""

from typing import Optional, Any
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
    companies: list[dict] = Field(default_factory=list)
    activities: list[dict] = Field(default_factory=list)
    opportunities: list[dict] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    renewals: list[dict] = Field(default_factory=list)
    pipeline_summary: Optional[dict] = None


# =============================================================================
# Meta Info
# =============================================================================

class MetaInfo(BaseModel):
    """Metadata about the response."""
    mode_used: str  # "docs", "data", "data+docs"
    latency_ms: int
    company_id: Optional[str] = None
    days: Optional[int] = None


# =============================================================================
# Request/Response
# =============================================================================

class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""
    question: str
    mode: Optional[str] = "auto"  # "auto", "docs", "data", "data+docs"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    company_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response to the frontend."""
    answer: str
    sources: list[Source]
    steps: list[Step]
    raw_data: RawData
    meta: MetaInfo


# =============================================================================
# Router Models
# =============================================================================

class RouterResult(BaseModel):
    """Result from the router's analysis."""
    mode_used: str  # "docs", "data", "data+docs"
    company_id: Optional[str] = None
    company_name_query: Optional[str] = None  # If we need to resolve a name
    days: int = 90
    intent: str = "general"  # "company_status", "renewals", "pipeline", "docs", "general"


# =============================================================================
# Tool Results
# =============================================================================

class ToolResult(BaseModel):
    """Result from a tool function."""
    data: Any
    sources: list[Source]
    error: Optional[str] = None
