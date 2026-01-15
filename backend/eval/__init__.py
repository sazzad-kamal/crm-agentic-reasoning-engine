"""Evaluation module for CRM AI agent.

Submodules:
- integration: Full conversation path evaluation (faithfulness, relevance, etc.)
- fetch: Node-level SQL + RAG evaluation
- shared: Common utilities (formatting, RAGAS metrics)

Usage:
    python -m backend.eval.integration --limit 5 -v
    python -m backend.eval.fetch --difficulty 1,2 --limit 10 -v
"""
