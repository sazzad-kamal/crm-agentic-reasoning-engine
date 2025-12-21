"""
Project 2: Agentic layer for Acme CRM AI Companion.

This package provides:
- CRM data access via DuckDB
- Tool functions for data retrieval
- Router for mode detection (docs/data/data+docs)
- Agent orchestration for answering questions
"""

from project2_agentic.agent import answer_question

__all__ = ["answer_question"]
