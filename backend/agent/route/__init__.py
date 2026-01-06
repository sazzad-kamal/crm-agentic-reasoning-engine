"""
Route node - classifies user questions and extracts entities.

Exports:
    route_node: LangGraph node for intent routing
"""

from backend.agent.route.node import route_node

__all__ = ["route_node"]
