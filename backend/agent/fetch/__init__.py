"""
Fetch node - retrieves CRM data based on routed intent.

Exports:
    fetch_node: LangGraph node for data fetching
    handlers: Intent-specific data handlers
"""

from backend.agent.fetch.node import fetch_node

__all__ = ["fetch_node"]
