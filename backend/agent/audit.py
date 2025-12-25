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
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.agent.config import get_config


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
    mode_used: str
    company_id: Optional[str] = None
    latency_ms: int = 0
    source_count: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class AgentAuditLogger:
    """
    Audit logger for agent queries.
    
    Writes structured JSONL entries for analysis and debugging.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
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
        mode_used: str,
        company_id: Optional[str] = None,
        latency_ms: int = 0,
        source_count: int = 0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log a query to the audit log.
        
        Args:
            question: The user's question
            mode_used: The routing mode used
            company_id: Resolved company ID (if any)
            latency_ms: Total latency in milliseconds
            source_count: Number of sources used
            user_id: User identifier (if available)
            session_id: Session identifier (if available)
            error: Error message (if query failed)
        """
        config = get_config()
        
        entry = AgentAuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            question=question[:500],  # Truncate long questions
            mode_used=mode_used,
            company_id=company_id,
            latency_ms=latency_ms,
            source_count=source_count,
            user_id=user_id,
            session_id=session_id,
            error=error,
        )
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
            logger.debug(f"Audit entry logged: mode={mode_used}, latency={latency_ms}ms")
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")
    
    def read_recent_queries(self, limit: int = 100) -> list[dict]:
        """
        Read recent queries from the audit log.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of audit entries (most recent first)
        """
        if not self.log_file.exists():
            return []
        
        entries = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            
            # Return most recent first
            return entries[-limit:][::-1]
        except Exception as e:
            logger.warning(f"Failed to read audit log: {e}")
            return []
    
    def get_stats(self) -> dict:
        """
        Get summary statistics from the audit log.
        
        Returns:
            Dictionary with query statistics
        """
        entries = self.read_recent_queries(limit=1000)
        
        if not entries:
            return {"total_queries": 0}
        
        from collections import Counter
        modes = Counter(e.get("mode_used", "unknown") for e in entries)
        latencies = [e["latency_ms"] for e in entries if e.get("latency_ms")]
        errors = sum(1 for e in entries if e.get("error"))
        
        return {
            "total_queries": len(entries),
            "modes": modes,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "error_rate": errors / len(entries) if entries else 0,
        }


# Module-level convenience functions
_logger_instance: Optional[AgentAuditLogger] = None


def get_audit_logger() -> AgentAuditLogger:
    """Get the global audit logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AgentAuditLogger()
    return _logger_instance
