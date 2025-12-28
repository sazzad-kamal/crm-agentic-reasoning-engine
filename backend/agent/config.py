"""
Centralized configuration for the Agentic layer.

All constants, model names, and tunable parameters are defined here.
Import from this module instead of defining constants in individual files.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Configure module logger
logger = logging.getLogger(__name__)

# Resolve backend root directory (for default paths)
_BACKEND_ROOT = Path(__file__).parent.parent.resolve()
_DEFAULT_CSV_DIR = _BACKEND_ROOT / "data" / "csv"
_DEFAULT_AUDIT_LOG = _BACKEND_ROOT / "data" / "logs" / "agent_audit.jsonl"


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
        description="LLM model for agent orchestration and synthesis (upgraded from gpt-4o)"
    )
    router_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for routing decisions (fast, cheap)"
    )
    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature for agent responses"
    )
    router_temperature: float = Field(
        default=0.0,
        description="LLM temperature for routing (0 for deterministic)"
    )
    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens in LLM response"
    )
    
    # -------------------------------------------------------------------------
    # Router Configuration
    # -------------------------------------------------------------------------
    use_llm_router: bool = Field(
        default=True,
        description="Use LLM for routing instead of heuristics"
    )
    fallback_to_heuristics: bool = Field(
        default=True,
        description="Fall back to heuristics if LLM routing fails"
    )
    
    # -------------------------------------------------------------------------
    # Feature Flags
    # -------------------------------------------------------------------------
    enable_follow_up_suggestions: bool = Field(
        default=True,
        description="Generate follow-up question suggestions"
    )
    enable_docs_integration: bool = Field(
        default=True,
        description="Include RAG docs in agent responses"
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging for agent queries"
    )

    # -------------------------------------------------------------------------
    # RAG Pipeline Configuration
    # -------------------------------------------------------------------------
    rag_use_hyde: bool = Field(
        default=False,
        description="Use HyDE (Hypothetical Document Embeddings) for retrieval - adds ~2-3s latency"
    )
    rag_use_rewrite: bool = Field(
        default=False,
        description="Use query rewriting for better retrieval - adds ~1-2s latency"
    )
    rag_use_reranker: bool = Field(
        default=True,
        description="Use reranker for better result ordering"
    )
    
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    csv_dir: Path = Field(
        default=_DEFAULT_CSV_DIR,
        description="Path to CSV data directory"
    )
    audit_log_file: Path = Field(
        default=_DEFAULT_AUDIT_LOG,
        description="Path to agent audit log file"
    )
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        extra="ignore",
    )
    
    @model_validator(mode='after')
    def validate_config(self) -> 'AgentConfig':
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
    """Validate configuration at startup and log warnings."""
    issues = []
    
    # Check if OpenAI API key is set
    if not os.environ.get('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not set - LLM calls will fail")
        issues.append("OPENAI_API_KEY not set")
    
    # Check if CSV directory exists
    if not config.csv_dir.exists():
        logger.warning(f"CSV directory not found: {config.csv_dir}")
        issues.append(f"CSV directory not found: {config.csv_dir}")
    
    # Create audit log directory if needed
    if config.enable_audit_logging:
        audit_dir = config.audit_log_file.parent
        if not audit_dir.exists():
            logger.info(f"Creating audit log directory: {audit_dir}")
            audit_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration summary
    logger.info(
        f"Agent Config loaded: llm_model={config.llm_model}, "
        f"use_llm_router={config.use_llm_router}, "
        f"follow_up_suggestions={config.enable_follow_up_suggestions}"
    )
    
    if issues:
        logger.warning(f"Configuration issues: {issues}")


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None


# Convenience exports
def is_mock_mode() -> bool:
    """Check if we're in mock mode (for testing)."""
    return os.environ.get("MOCK_LLM", "0") == "1"
