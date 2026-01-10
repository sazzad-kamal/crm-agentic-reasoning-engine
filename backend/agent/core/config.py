"""
Centralized configuration for the Agentic layer.

All constants, model names, and tunable parameters are defined here.
Import from this module instead of defining constants in individual files.
"""

import logging
import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure module logger
logger = logging.getLogger(__name__)

# Resolve backend root directory (for default paths)
# backend/agent/core/config.py -> core -> agent -> backend
_BACKEND_ROOT = Path(__file__).parent.parent.parent.resolve()
_DEFAULT_AUDIT_LOG = _BACKEND_ROOT / "eval" / "output" / "audit.jsonl"


class AgentConfig(BaseSettings):
    """
    Agentic layer configuration with environment variable support.

    All settings can be overridden via environment variables with the
    AGENT_ prefix (e.g., AGENT_LLM_MODEL=gpt-4).
    """

    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    llm_model: str = Field(
        default="gpt-5.2",
        description="LLM model for agent orchestration and synthesis (upgraded from gpt-4o)",
    )
    router_model: str = Field(
        default="gpt-4o-mini", description="LLM model for routing decisions (fast, cheap)"
    )
    llm_temperature: float = Field(default=0.1, description="LLM temperature for agent responses")
    llm_max_tokens: int = Field(default=1024, description="Maximum tokens in LLM response")

    # -------------------------------------------------------------------------
    # Feature Flags
    # -------------------------------------------------------------------------
    enable_follow_up_suggestions: bool = Field(
        default=True, description="Generate follow-up question suggestions"
    )

    # -------------------------------------------------------------------------
    # Node Configuration
    # -------------------------------------------------------------------------
    default_days: int = Field(
        default=90, description="Default time window in days for data queries"
    )
    max_followup_suggestions: int = Field(
        default=3, description="Maximum number of follow-up suggestions to return"
    )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    audit_log_file: Path = Field(
        default=_DEFAULT_AUDIT_LOG, description="Path to agent audit log file"
    )

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "AgentConfig":
        """Cross-field validation."""
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("llm_temperature must be between 0 and 2")
        return self


# Global config instance (lazy initialization)
_config: AgentConfig | None = None


def get_config() -> AgentConfig:
    """Get the global Agent configuration instance with validation."""
    global _config
    if _config is None:
        _config = AgentConfig()
        _validate_startup_config(_config)
    return _config


def _validate_startup_config(config: AgentConfig) -> None:
    """Validate configuration at startup."""
    # Log configuration summary
    logger.info(
        f"Agent Config loaded: llm_model={config.llm_model}, "
        f"router_model={config.router_model}, "
        f"follow_up_suggestions={config.enable_follow_up_suggestions}"
    )


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None


# Convenience exports
def is_mock_mode() -> bool:
    """Check if we're in mock mode (for testing)."""
    return os.environ.get("MOCK_LLM", "0") == "1"


__all__ = [
    "AgentConfig",
    "get_config",
    "reset_config",
    "is_mock_mode",
]
