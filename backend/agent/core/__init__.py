"""
Core agent types, configuration, and state.

This module provides foundational types, configuration,
and state definitions for the agent system.
"""

from backend.agent.core.config import (
    AgentConfig,
    get_config,
    is_mock_mode,
    reset_config,
)
from backend.agent.core.state import AgentState, Message, Source, format_history_for_prompt

__all__ = [
    # Config
    "AgentConfig",
    "get_config",
    "reset_config",
    "is_mock_mode",
    # Types
    "Source",
    # State
    "AgentState",
    "Message",
    "format_history_for_prompt",
]
