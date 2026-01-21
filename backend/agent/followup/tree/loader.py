"""Question tree loader and graph builder."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

__all__ = ["get_starters", "get_follow_ups", "get_graph"]

# Load JSON and build graph
_DATA_PATH = Path(__file__).parent / "data.json"
with open(_DATA_PATH) as f:
    _raw_data: dict[str, list[str]] = json.load(f)

_STARTERS: list[str] = [
    "What deals are in the pipeline?",
    "Which accounts are at risk?",
    "Which contacts haven't been contacted recently?",
]

_G = nx.DiGraph()
for question, follow_ups in _raw_data.items():
    _G.add_node(question)
    for follow_up in follow_ups:
        _G.add_edge(question, follow_up)


def get_starters() -> list[str]:
    """Get the starter questions."""
    return _STARTERS.copy()


def get_follow_ups(question: str) -> list[str]:
    """Get follow-up questions for a given question (up to 3)."""
    if question in _G:
        return list(_G.successors(question))[:3]
    return []


def get_graph() -> nx.DiGraph:
    """Get the question tree graph (read-only copy)."""
    return _G.copy()
