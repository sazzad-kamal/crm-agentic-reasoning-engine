"""
Answer node - synthesizes natural language responses from CRM data.

Exports:
    answer_node: LangGraph node for answer generation
    formatters: Data section formatters for LLM prompts
"""

from backend.agent.answer.node import answer_node

__all__ = ["answer_node"]
