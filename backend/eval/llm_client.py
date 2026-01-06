"""
Simple LLM client for evaluation harnesses.

Uses the shared backend/llm infrastructure.
"""

from backend.llm.client import call_llm


__all__ = ["call_llm"]
