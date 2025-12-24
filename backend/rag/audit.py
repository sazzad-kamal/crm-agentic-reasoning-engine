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
from typing import Optional, Any

from backend.rag.config import get_config


# Configure module logger
logger = logging.getLogger(__name__)

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
    config = get_config()
    
    try:
        log_file = config.audit_log_path
        
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe file write
        with _audit_lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        
        logger.debug(f"Audit logged: query='{entry.query[:30]}...', status={entry.status}")
        
    except Exception as e:
        # Don't fail the request if audit logging fails
        logger.error(f"Failed to write audit log: {e}")


def read_audit_log(
    limit: int = 100,
    company_id: Optional[str] = None,
    since: Optional[datetime] = None,
) -> list[dict]:
    """
    Read entries from the audit log.
    
    Args:
        limit: Maximum number of entries to return
        company_id: Filter by company_id
        since: Filter entries after this timestamp
        
    Returns:
        List of audit entries as dictionaries
    """
    config = get_config()
    log_file = config.audit_log_path
    
    if not log_file.exists():
        return []
    
    entries = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    
                    # Apply filters
                    if company_id and entry.get("company_id") != company_id:
                        continue
                    if since:
                        entry_time = datetime.fromisoformat(entry["timestamp"].rstrip("Z"))
                        if entry_time < since:
                            continue
                    
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        # Return most recent entries first
        return entries[-limit:][::-1]
        
    except Exception as e:
        logger.error(f"Failed to read audit log: {e}")
        return []


def get_audit_stats(company_id: Optional[str] = None) -> dict:
    """
    Get aggregate statistics from the audit log.
    
    Args:
        company_id: Filter by company_id (optional)
        
    Returns:
        Dictionary with usage statistics
    """
    entries = read_audit_log(limit=10000, company_id=company_id)
    
    if not entries:
        return {
            "total_queries": 0,
            "success_rate": 0.0,
            "avg_latency_ms": 0,
            "unique_users": 0,
            "unique_companies": 0,
        }
    
    total = len(entries)
    successes = sum(1 for e in entries if e.get("status") == "success")
    latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms")]
    users = set(e.get("user_id") for e in entries if e.get("user_id"))
    companies = set(e.get("company_id") for e in entries if e.get("company_id"))
    
    return {
        "total_queries": total,
        "success_rate": successes / total if total > 0 else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "unique_users": len(users),
        "unique_companies": len(companies),
    }
