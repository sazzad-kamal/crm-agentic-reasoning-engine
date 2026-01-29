"""Question tree loader and graph builder."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import networkx as nx

__all__ = ["get_starters", "get_follow_ups", "get_graph"]

_DATA_PATH = Path(__file__).parent / "data.json"

_STARTERS = (
    "What deals are in the pipeline?",
    "Which accounts are up for renewal?",
    "What tasks are due this week?",
)


@cache
def _load_graph() -> nx.DiGraph:
    """Lazily load and build the question graph."""
    with open(_DATA_PATH) as f:
        raw_data: dict[str, list[str]] = json.load(f)

    g = nx.DiGraph()
    for question, follow_ups in raw_data.items():
        for follow_up in follow_ups:
            g.add_edge(question, follow_up)
    return g


def get_starters() -> list[str]:
    """Get the starter questions."""
    return list(_STARTERS)


def get_follow_ups(question: str) -> list[str]:
    """Get follow-up questions for a given question."""
    g = _load_graph()
    if question in g:
        return list(g.successors(question))
    return []


def get_graph() -> nx.DiGraph:
    """Get the question tree graph (read-only copy)."""
    return _load_graph().copy()
