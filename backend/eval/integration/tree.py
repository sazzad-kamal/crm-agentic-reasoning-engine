"""
Evaluation utilities for the question tree.

This module provides functions for:
- Loading expected answers and RAG flags from fixtures
- Generating test paths from the question tree
- Computing tree statistics

The question tree data itself lives in backend/agent/followup/tree/
since runtime code (get_follow_ups) needs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

# Import the shared graph from followup/tree
from backend.agent.followup.tree import _G, get_starters

if TYPE_CHECKING:
    from rich.tree import Tree

__all__ = [
    "get_expected_answer",
    "get_expected_rag",
    "get_all_paths",
    "get_tree_stats",
    "validate_tree",
    "print_tree",
]

# =============================================================================
# Load Expected Values (from eval/fixtures/)
# =============================================================================

_EVAL_FIXTURES_PATH = Path(__file__).parent / "fixtures"


def _load_yaml_fixture(filename: str) -> dict:
    """Load a YAML fixture file from eval/fixtures/."""
    filepath = _EVAL_FIXTURES_PATH / filename
    if filepath.exists():
        try:
            import yaml
            with open(filepath, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except Exception:
            return {}
    return {}


_EXPECTED_ANSWERS: dict[str, str] = _load_yaml_fixture("expected_answers.yaml")
_EXPECTED_RAG: dict[str, bool] = _load_yaml_fixture("expected_rag.yaml")

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


def get_expected_rag(question: str) -> bool | None:
    """
    Get the expected RAG decision for a question (for RAG decision accuracy).

    Args:
        question: The question to look up

    Returns:
        True if RAG should be invoked, False if not, None if not found
    """
    return _EXPECTED_RAG.get(question)


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


def validate_tree() -> list[str]:
    """
    Validate the question tree for consistency.

    Returns list of any issues found.
    """
    subgraph = _get_reachable_subgraph()
    reachable = set(subgraph.nodes())
    issues = []

    # Check all starters are in tree
    for starter in _STARTERS:
        if starter not in _G:
            issues.append(f"Starter not in tree: {starter}")

    # Check for orphans
    for question in _G.nodes():
        if question not in reachable:
            issues.append(f"Orphaned question (not reachable): {question}")

    # Check subgraph is a valid DAG (no cycles)
    if not nx.is_directed_acyclic_graph(subgraph):
        issues.append("Tree contains cycles!")

    # Check each node has exactly 0 or 3 follow-ups (valid tree structure)
    for node in reachable:
        out_degree = _G.out_degree(node)
        if out_degree not in (0, 3):
            issues.append(f"Node has {out_degree} follow-ups (expected 0 or 3): {node[:50]}...")

    return issues


def print_tree(max_depth: int | None = None) -> Tree:
    """
    Generate a Rich Tree representation of the question tree.

    Args:
        max_depth: Maximum depth to display (default: show all levels)

    Returns:
        Rich Tree object (print with rich.print or console.print).

    Usage:
        from rich import print
        print(print_tree())              # Full tree
        print(print_tree(max_depth=3))   # 3 levels deep
    """
    from rich.tree import Tree

    # Entity labels for starters
    entity_labels = {
        "What deals are in the pipeline?": "[bold cyan]OPPORTUNITIES[/bold cyan]",
        "Which accounts are at risk?": "[bold green]COMPANIES[/bold green]",
        "Which contacts haven't been contacted recently?": "[bold yellow]CONTACTS[/bold yellow]",
    }

    def add_children(parent_branch: Tree, node: str, depth: int) -> None:
        """Recursively add children to tree."""
        if max_depth is not None and depth >= max_depth:
            return

        children = list(_G.successors(node)) if node in _G else []
        for child in children:
            child_branch = parent_branch.add(child)
            add_children(child_branch, child, depth + 1)

    # Create root tree
    root = Tree("[bold]Question Tree[/bold]")
    for starter in _STARTERS:
        label = entity_labels.get(starter, starter)
        entity_branch = root.add(label)
        starter_branch = entity_branch.add(f"[bold]{starter}[/bold]")
        add_children(starter_branch, starter, 1)

    return root
