"""Schemas for fetch handlers."""

from typing import Any

from pydantic import BaseModel

from backend.agent.core.state import Source


class ToolResult(BaseModel):
    """Result from a tool function."""

    data: dict[str, Any]
    sources: list[Source]
    error: str | None = None


__all__ = ["ToolResult"]
