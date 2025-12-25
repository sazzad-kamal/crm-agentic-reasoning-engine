"""
Shared pipeline utilities and base classes.

Contains common functionality used across different RAG pipelines:
- Progress tracking
- Context building
- Query preprocessing

NOTE: Context building has been consolidated into backend.common.context_builder.
This module re-exports those functions for backwards compatibility.
"""

import logging
import time
from typing import Optional, Callable
from collections import defaultdict

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.pipeline.constants import (
    MAX_CONTEXT_TOKENS,
    MIN_BM25_SCORE_RATIO,
    MAX_CHUNKS_PER_DOC,
    MAX_CHUNKS_PER_TYPE,
)
from backend.rag.pipeline.utils import estimate_tokens, tokens_to_chars

# Re-export gating functions for backwards compatibility
from backend.rag.pipeline.gating import (
    apply_lexical_gate,
    apply_per_doc_cap,
    apply_per_type_cap,
    apply_all_gates,
)

# Re-export context builders from common module
from backend.common.context_builder import (
    build_context,
    build_context_with_sources,
    build_private_context,
    build_docs_context,
    ContextBuilder,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Progress Tracking
# =============================================================================

class PipelineProgress:
    """
    Tracks and logs pipeline step progress.
    
    Useful for UI progress indicators and debugging.
    """
    
    def __init__(self, callback: Optional[Callable[[str, str, float], None]] = None):
        """
        Initialize progress tracker.
        
        Args:
            callback: Optional function called with (step_id, label, elapsed_ms)
        """
        self.steps: list[dict] = []
        self.callback = callback
        self._start_time = time.time()
        self._step_start: Optional[float] = None
    
    def start_step(self, step_id: str, label: str) -> None:
        """Start tracking a new step."""
        self._step_start = time.time()
        logger.info(f"[STEP] Starting: {label}")
        if self.callback:
            self.callback(step_id, f"Starting: {label}", 0)
    
    def complete_step(self, step_id: str, label: str, status: str = "done") -> None:
        """Mark a step as complete."""
        elapsed_ms = (time.time() - self._step_start) * 1000 if self._step_start else 0
        self.steps.append({
            "id": step_id,
            "label": label,
            "status": status,
            "elapsed_ms": elapsed_ms,
        })
        logger.info(f"[STEP] Completed: {label} ({elapsed_ms:.0f}ms) - {status}")
        if self.callback:
            self.callback(step_id, label, elapsed_ms)
    
    def get_steps(self) -> list[dict]:
        """Get all completed steps."""
        return self.steps
    
    def total_elapsed_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return (time.time() - self._start_time) * 1000


# NOTE: Context building functions have been moved to backend.common.context_builder
# They are re-exported at the top of this module for backwards compatibility.
