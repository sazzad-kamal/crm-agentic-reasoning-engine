"""
Audit logging for the Agentic layer.

Provides structured logging for:
- Query history
- Routing decisions
- Response metadata
- Error tracking

Logs are written to JSONL format for easy analysis.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.agent.core.config import get_config

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class AgentAuditEntry:
    """
    Audit entry for an agent query.

    Captures key metadata for observability and debugging.
    """

    timestamp: str
    question: str
    company_id: str | None = None
    latency_ms: int = 0
    source_count: int = 0
    user_id: str | None = None
    session_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class AgentAuditLogger:
    """
    Audit logger for agent queries.

    Writes structured JSONL entries for analysis and debugging.
    """

    def __init__(self, log_file: Path | None = None) -> None:
        """Initialize the audit logger."""
        config = get_config()
        self.log_file = log_file or config.audit_log_file
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Ensure the log directory exists."""
        log_dir = self.log_file.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created audit log directory: {log_dir}")

    def log_query(
        self,
        question: str,
        company_id: str | None = None,
        latency_ms: int = 0,
        source_count: int = 0,
        user_id: str | None = None,
        session_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """
        Log a query to the audit log.

        Args:
            question: The user's question
            company_id: Resolved company ID (if any)
            latency_ms: Total latency in milliseconds
            source_count: Number of sources used
            user_id: User identifier (if available)
            session_id: Session identifier (if available)
            error: Error message (if query failed)
        """
        entry = AgentAuditEntry(
            timestamp=datetime.now(UTC).isoformat(),
            question=question[:500],  # Truncate long questions
            company_id=company_id,
            latency_ms=latency_ms,
            source_count=source_count,
            user_id=user_id,
            session_id=session_id,
            error=error,
        )

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(entry.to_dict())}\n")
            logger.debug(f"Audit entry logged: latency={latency_ms}ms")
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")


# Module-level convenience functions
_logger_instance: AgentAuditLogger | None = None


def get_audit_logger() -> AgentAuditLogger:
    """Get the global audit logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AgentAuditLogger()
    return _logger_instance


__all__ = [
    "AgentAuditEntry",
    "AgentAuditLogger",
    "get_audit_logger",
]
