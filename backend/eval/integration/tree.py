"""Evaluation utilities for the question tree."""

from __future__ import annotations

import logging
from functools import cache

import networkx as nx

from backend.agent.followup.tree import get_graph, get_starters

logger = logging.getLogger(__name__)


# --- Expected fixtures ---


@cache
def _load_expected() -> dict[str, dict]:
    """Load expected fixtures from shared questions (cached)."""
    from backend.eval.answer.shared.loader import load_questions

    try:
        questions = load_questions()
        return {
            q.text: {"answer": q.expected_answer or None, "action": q.expected_action}
            for q in questions
        }
    except Exception as e:
        logger.warning(f"Failed to load questions: {e}")
        return {}


def get_expected_answer(question: str) -> str | None:
    """Get the expected answer for a question (for RAGAS answer_correctness)."""
    return _load_expected().get(question, {}).get("answer")


def get_expected_action(question: str) -> bool | None:
    """Get whether an action is expected for a question. None if not in fixture."""
    return _load_expected().get(question, {}).get("action")


# --- Tree paths ---


def _find_paths(graph: nx.DiGraph, starters: list[str]) -> list[list[str]]:
    """Find all paths from starters to leaf nodes via DFS."""
    paths: list[list[str]] = []

    def _dfs(node: str, path: list[str]) -> None:
        if graph.out_degree(node) == 0:
            paths.append(path)
            return
        for child in graph.successors(node):
            _dfs(child, path + [child])

    for starter in starters:
        _dfs(starter, [starter])
    return paths


@cache
def _compute_paths_and_stats() -> tuple[list[list[str]], dict]:
    """Compute all paths and tree stats in a single pass (cached)."""
    graph = get_graph()
    starters = [s for s in get_starters() if s in graph]
    paths = _find_paths(graph, starters)

    stats = {
        "num_starters": len(starters),
        "num_questions": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_paths": len(paths),
        "path_lengths": {
            "min": min(len(p) for p in paths) if paths else 0,
            "max": max(len(p) for p in paths) if paths else 0,
        },
    }
    return paths, stats


def get_all_paths() -> list[list[str]]:
    """Get all conversation paths from starters to leaf nodes."""
    paths, _ = _compute_paths_and_stats()
    return paths


def get_tree_stats() -> dict:
    """Get statistics about the question tree."""
    _, stats = _compute_paths_and_stats()
    return stats


__all__ = [
    "get_all_paths",
    "get_expected_action",
    "get_expected_answer",
    "get_tree_stats",
]
