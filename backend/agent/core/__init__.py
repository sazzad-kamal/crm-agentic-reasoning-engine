"""
Core agent types and configuration.

This module provides foundational types, schemas, and configuration
for the agent system.
"""

from backend.agent.core.config import (
    AgentConfig,
    get_config,
    reset_config,
    is_mock_mode,
)
from backend.agent.core.schemas import (
    Source,
    RawData,
    MetaInfo,
    ChatResponse,
    RouterResult,
    ToolResult,
)

__all__ = [
    # Config
    "AgentConfig",
    "get_config",
    "reset_config",
    "is_mock_mode",
    # Schemas
    "Source",
    "RawData",
    "MetaInfo",
    "ChatResponse",
    "RouterResult",
    "ToolResult",
]
