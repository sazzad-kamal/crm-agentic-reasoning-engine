"""
Core agent types, configuration, and state.

This module provides foundational types, schemas, configuration,
state definitions, and memory utilities for the agent system.
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
from backend.agent.core.state import AgentState, Message
from backend.agent.core.memory import (
    clear_session,
    format_history_for_prompt,
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
    # State
    "AgentState",
    "Message",
    # Memory
    "clear_session",
    "format_history_for_prompt",
]
