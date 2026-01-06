"""Schemas for the route module."""

from pydantic import BaseModel, Field


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


__all__ = ["RouterResult"]
