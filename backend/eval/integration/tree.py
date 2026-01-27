"""
Evaluation utilities for the question tree.

This module provides functions for:
- Loading expected answers from fixtures
- Generating test paths from the question tree
- Computing tree statistics
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

# Import the shared graph from followup/tree via public API
from backend.agent.followup.tree import get_graph, get_starters

__all__ = [
    "get_expected_answer",
    "get_all_paths",
    "get_tree_stats",
]

# Get a copy of the graph (read-only)
_G = get_graph()

# =============================================================================
# Load Expected Values (from eval/fixtures/)
# =============================================================================

_EVAL_FIXTURES_PATH = Path(__file__).parent / "fixtures"


def _load_yaml_fixture(filename: str) -> dict:
    """Load a YAML fixture file from eval/fixtures/."""
    import logging

    filepath = _EVAL_FIXTURES_PATH / filename
    if not filepath.exists():
        return {}
    try:
        import yaml

        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load {filename}: {e}")
        return {}


_EXPECTED_ANSWERS: dict[str, str] = _load_yaml_fixture("expected_answers.yaml")

# Cache starters for reuse
_STARTERS = get_starters()


# =============================================================================
# Internal Helpers
# =============================================================================


def _compute_max_depth(starters: list[str]) -> int:
    """Compute the maximum depth from any starter to any descendant."""
    max_depth = 0
    for starter in starters:
        if starter in _G:
            for node in nx.descendants(_G, starter):
                try:
                    path_len = nx.shortest_path_length(_G, starter, node) + 1
                    max_depth = max(max_depth, path_len)
                except nx.NetworkXNoPath:
                    pass
    return max_depth


def _find_paths(starters: list[str], subgraph: nx.DiGraph, max_depth: int) -> list[list[str]]:
    """Find all paths from starters to leaf nodes."""
    leaves = [n for n in subgraph.nodes() if subgraph.out_degree(n) == 0]
    paths: list[list[str]] = []
    for starter in starters:
        for leaf in leaves:
            try:
                for path in nx.all_simple_paths(_G, starter, leaf, cutoff=max_depth - 1):
                    paths.append(path)
            except nx.NetworkXNoPath:
                continue
    return paths


def _get_reachable_subgraph() -> nx.DiGraph:
    """Get subgraph of all nodes reachable from starters."""
    reachable: set[str] = set()
    for starter in _STARTERS:
        if starter in _G:
            reachable.add(starter)
            reachable |= nx.descendants(_G, starter)
    return _G.subgraph(reachable)


# =============================================================================
# Expected Values API
# =============================================================================


def get_expected_answer(question: str) -> str | None:
    """
    Get the expected answer for a question (for RAGAS answer_correctness).

    Args:
        question: The question to look up

    Returns:
        Expected answer string, or None if not found
    """
    return _EXPECTED_ANSWERS.get(question)


# =============================================================================
# Path Generation & Tree Stats
# =============================================================================


def get_all_paths() -> list[list[str]]:
    """
    Get all conversation paths from starters to leaf nodes.

    Returns:
        List of paths, where each path is a list of questions from starter to terminal.
    """
    subgraph = _get_reachable_subgraph()
    max_depth = _compute_max_depth(_STARTERS)
    return _find_paths(_STARTERS, subgraph, max_depth)


def get_tree_stats() -> dict:
    """Get statistics about the question tree."""
    subgraph = _get_reachable_subgraph()
    max_depth = _compute_max_depth(_STARTERS)
    paths = _find_paths(_STARTERS, subgraph, max_depth)

    return {
        "num_starters": len(_STARTERS),
        "num_questions": subgraph.number_of_nodes(),
        "num_edges": subgraph.number_of_edges(),
        "num_paths": len(paths),
        "max_depth": max_depth,
        "path_lengths": {
            "min": min(len(p) for p in paths) if paths else 0,
            "max": max(len(p) for p in paths) if paths else 0,
        },
    }


