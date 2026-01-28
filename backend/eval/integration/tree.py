"""Evaluation utilities for the question tree."""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import networkx as nx
import yaml

from backend.agent.followup.tree import get_graph, get_starters

logger = logging.getLogger(__name__)

_EVAL_FIXTURES_PATH = Path(__file__).parent / "fixtures"


@cache
def _get_graph() -> nx.DiGraph:
    return get_graph()


@cache
def _get_starters() -> list[str]:
    return get_starters()


def _load_yaml_fixture(filename: str) -> dict:
    """Load a YAML fixture file (cached per filename)."""
    filepath = _EVAL_FIXTURES_PATH / filename
    if not filepath.exists():
        return {}
    try:
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except Exception as e:
        logger.warning(f"Failed to load {filename}: {e}")
        return {}


@cache
def _load_expected_answers() -> dict[str, str]:
    """Load expected answers from YAML fixture (cached)."""
    return _load_yaml_fixture("expected_answers.yaml")


@cache
def _load_expected_actions() -> dict[str, bool]:
    """Load expected action flags from YAML fixture (cached)."""
    return _load_yaml_fixture("expected_actions.yaml")


def _compute_max_depth(graph: nx.DiGraph, starters: list[str]) -> int:
    """Compute the maximum depth (node count) from any starter to its deepest descendant."""
    max_depth = 0
    for starter in starters:
        if starter in graph:
            lengths = nx.single_source_shortest_path_length(graph, starter)
            descendant_lengths = [d for node, d in lengths.items() if node != starter]
            if descendant_lengths:
                max_depth = max(max_depth, max(descendant_lengths) + 1)
    return max_depth


def _find_paths(graph: nx.DiGraph, starters: list[str], max_depth: int) -> list[list[str]]:
    """Find all paths from starters to leaf nodes in the graph."""
    leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    paths: list[list[str]] = []
    for starter in starters:
        for leaf in leaves:
            try:
                for path in nx.all_simple_paths(graph, starter, leaf, cutoff=max_depth - 1):
                    paths.append(path)
            except nx.NetworkXNoPath:
                continue
    return paths


@cache
def _compute_paths_and_stats() -> tuple[list[list[str]], dict]:
    """Compute all paths and tree stats in a single pass (cached)."""
    graph = _get_graph()
    starters = _get_starters()

    # Build reachable subgraph
    reachable: set[str] = set()
    for starter in starters:
        if starter in graph:
            reachable.add(starter)
            reachable |= nx.descendants(graph, starter)
    subgraph = graph.subgraph(reachable).copy()

    max_depth = _compute_max_depth(subgraph, starters)
    paths = _find_paths(subgraph, starters, max_depth)

    stats = {
        "num_starters": len(starters),
        "num_questions": subgraph.number_of_nodes(),
        "num_edges": subgraph.number_of_edges(),
        "num_paths": len(paths),
        "max_depth": max_depth,
        "path_lengths": {
            "min": min(len(p) for p in paths) if paths else 0,
            "max": max(len(p) for p in paths) if paths else 0,
        },
    }
    return paths, stats


def get_expected_answer(question: str) -> str | None:
    """Get the expected answer for a question (for RAGAS answer_correctness)."""
    return _load_expected_answers().get(question)


def get_expected_action(question: str) -> bool | None:
    """Get whether an action is expected for a question. None if not in fixture."""
    return _load_expected_actions().get(question)


def get_all_paths() -> list[list[str]]:
    """Get all conversation paths from starters to leaf nodes."""
    paths, _ = _compute_paths_and_stats()
    return paths


def get_tree_stats() -> dict:
    """Get statistics about the question tree."""
    _, stats = _compute_paths_and_stats()
    return stats
