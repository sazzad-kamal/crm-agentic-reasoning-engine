"""
Followup node - generates follow-up question suggestions.

Exports:
    followup_node: LangGraph node for follow-up suggestions
    tree: Hardcoded question tree for demo reliability
"""

from backend.agent.followup.node import followup_node

__all__ = ["followup_node"]
