"""
Audit logging for RAG pipeline queries.

Logs all queries with timestamps, company_id, latency, and results for:
- Compliance and debugging
- Usage analytics
- Performance monitoring
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


# Configure module logger
logger = logging.getLogger(__name__)

# Audit log path
_BACKEND_ROOT = Path(__file__).parent.parent
AUDIT_LOG_PATH = _BACKEND_ROOT / "data/logs/audit.jsonl"

# Thread-safe file writing lock
_audit_lock = threading.Lock()


class AuditEntry:
    """Represents a single audit log entry."""
    
    def __init__(
        self,
        query: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        mode: str = "auto",
    ):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.query = query
        self.company_id = company_id
        self.user_id = user_id
        self.session_id = session_id
        self.mode = mode
        
        # Filled in after processing
        self.rewritten_query: Optional[str] = None
        self.num_chunks_retrieved: int = 0
        self.num_chunks_used: int = 0
        self.answer_length: int = 0
        self.latency_ms: int = 0
        self.status: str = "pending"  # pending, success, error
        self.error_message: Optional[str] = None
        self.sources: list[str] = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "company_id": self.company_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "mode": self.mode,
            "num_chunks_retrieved": self.num_chunks_retrieved,
            "num_chunks_used": self.num_chunks_used,
            "answer_length": self.answer_length,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error_message": self.error_message,
            "sources": self.sources,
        }


def log_audit_entry(entry: AuditEntry) -> None:
    """
    Write an audit entry to the audit log file.
    
    Thread-safe and handles file errors gracefully.
    """
    try:
        # Ensure directory exists
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe file write
        with _audit_lock:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        
        logger.debug(f"Audit logged: query='{entry.query[:30]}...', status={entry.status}")
        
    except Exception as e:
        # Don't fail the request if audit logging fails
        logger.error(f"Failed to write audit log: {e}")
