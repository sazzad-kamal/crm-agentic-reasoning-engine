"""
Hardcoded question tree for demo reliability.

This module provides a deterministic tree of questions and follow-ups,
ensuring 100% reliability for demos - no LLM generation variability.

Structure:
- 3 role-based starter questions:
  * Sales Rep (jsmith): "How's my pipeline?" - Pipeline focus
  * CSM (amartin): "Any renewals at risk?" - Retention focus
  * Manager: "How's the team doing?" - Aggregate view
- Each question has 3 follow-ups, with varying depths (4-6 levels)
- Run `python -m backend.agent.followup.tree stats` for current metrics

Usage:
    from backend.agent.followup.tree import get_follow_ups, get_starters, get_paths_for_role

    # Get starter questions
    starters = get_starters()

    # Get follow-ups for a question
    follow_ups = get_follow_ups("How's my pipeline?")

    # Get all paths for a role (or all roles if None)
    paths = get_paths_for_role("sales")

CLI:
    python -m backend.agent.followup.tree validate [--role sales|csm|manager]
    python -m backend.agent.followup.tree tree [--role sales|csm|manager] [--depth N]
    python -m backend.agent.followup.tree stats [--role sales|csm|manager]
    python -m backend.agent.followup.tree paths --role sales [--limit N]
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from rich.tree import Tree

__all__ = [
    "get_starters",
    "get_follow_ups",
    "get_paths_for_role",
    "get_tree_stats",
    "validate_tree",
    "print_tree",
    "get_expected_answer",
    "get_expected_rag",
    "get_expected_sql_results",
    "validate_sql_results",
]

# =============================================================================
# Load JSON and Build Graph
# =============================================================================

_DATA_PATH = Path(__file__).parent / "data.json"
with open(_DATA_PATH) as f:
    _raw_data: dict[str, list[str]] = json.load(f)

# =============================================================================
# Load Expected Values (from eval/fixtures/)
# =============================================================================

_EVAL_FIXTURES_PATH = Path(__file__).parents[3] / "eval" / "fixtures"


def _load_yaml_fixture(filename: str) -> dict:
    """Load a YAML fixture file from eval/fixtures/."""
    filepath = _EVAL_FIXTURES_PATH / filename
    if filepath.exists():
        try:
            import yaml  # type: ignore[import-untyped]
            with open(filepath, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except Exception:
            return {}
    return {}


_EXPECTED_ANSWERS: dict[str, str] = _load_yaml_fixture("expected_answers.yaml")
_EXPECTED_RAG: dict[str, bool] = _load_yaml_fixture("expected_rag.yaml")
_EXPECTED_SQL_RESULTS: dict[str, dict] = _load_yaml_fixture("expected_sql_results.yaml")

# Role mapping - starters are derived from this
_ROLE_MAP = {
    "sales": "Show my open deals",
    "csm": "Show my at-risk renewals",
    "manager": "Show team pipeline",
}
_STARTERS: list[str] = list(_ROLE_MAP.values())

# Build directed graph
_G = nx.DiGraph()

for question, follow_ups in _raw_data.items():
    _G.add_node(question)
    for follow_up in follow_ups:
        _G.add_edge(question, follow_up)


# =============================================================================
# Internal Helpers
# =============================================================================


def _get_starters_for_role(role: str | None) -> list[str]:
    """Get starter questions for a role. None means all roles."""
    if role is None:
        return _STARTERS.copy()
    role_lower = role.lower()
    if role_lower not in _ROLE_MAP:
        raise ValueError(f"Unknown role: {role}. Use: sales, csm, or manager")
    return [_ROLE_MAP[role_lower]]


def _get_subgraph(role: str | None) -> tuple[list[str], nx.DiGraph]:
    """Get starters and subgraph for a role. None means all roles."""
    starters = _get_starters_for_role(role)
    role_nodes: set[str] = set()
    for starter in starters:
        if starter in _G:
            role_nodes.add(starter)
            role_nodes |= nx.descendants(_G, starter)
    return starters, _G.subgraph(role_nodes)


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


# =============================================================================
# Public API
# =============================================================================


def get_starters() -> list[str]:
    """Get the starter questions."""
    return _STARTERS.copy()


def get_follow_ups(question: str) -> list[str]:
    """
    Get follow-up questions for a given question.

    Returns hardcoded follow-ups from the tree, or empty list
    if the question isn't in the tree.
    """
    if question in _G:
        return list(_G.successors(question))
    return []


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


def get_expected_sql_results(question: str) -> dict | None:
    """
    Get the expected SQL results assertions for a question.

    Args:
        question: The question to look up

    Returns:
        Dict of assertions (row_count_min, total_value, must_contain, etc.), or None if not found
    """
    return _EXPECTED_SQL_RESULTS.get(question)


def validate_sql_results(question: str, sql_results: dict) -> tuple[bool, list[str]]:
    """
    Validate SQL query results against expected assertions.

    Args:
        question: The question that was asked
        sql_results: Dict of query name -> list of result rows

    Returns:
        Tuple of (passed: bool, errors: list[str])
        - passed: True if all assertions pass, False if any fail
        - errors: List of error messages describing failures

    Assertion types:
        row_count: Exact number of rows expected
        row_count_min: Minimum number of rows
        row_count_max: Maximum number of rows
        total_value: Sum of value columns (10% tolerance)
        must_contain: Value(s) must appear in results
        must_contain_all: All values must appear
        must_contain_any: At least one value must appear
        must_not_contain: Value(s) must NOT appear in results
        exact_values: List of exact values that must match a column
    """
    expected = get_expected_sql_results(question)
    if expected is None:
        # No assertions defined for this question
        return True, []

    errors: list[str] = []

    # Flatten all results into a single list of rows and a string for text search
    all_rows: list[dict] = []
    all_values: list = []
    all_text = ""

    for _query_name, rows in sql_results.items():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    all_rows.append(row)
                    all_values.extend(row.values())
                    all_text += " ".join(str(v) for v in row.values()) + " "
                else:
                    all_values.append(row)
                    all_text += str(row) + " "
        elif isinstance(rows, dict):
            all_rows.append(rows)
            all_values.extend(rows.values())
            all_text += " ".join(str(v) for v in rows.values()) + " "

    all_text = all_text.lower()

    # Validate row_count (exact)
    if "row_count" in expected:
        exact_rows = expected["row_count"]
        if len(all_rows) != exact_rows:
            errors.append(f"row_count: got {len(all_rows)}, expected exactly {exact_rows}")

    # Validate row_count_min
    if "row_count_min" in expected:
        min_rows = expected["row_count_min"]
        if len(all_rows) < min_rows:
            errors.append(f"row_count: got {len(all_rows)}, expected >= {min_rows}")

    # Validate row_count_max
    if "row_count_max" in expected:
        max_rows = expected["row_count_max"]
        if len(all_rows) > max_rows:
            errors.append(f"row_count: got {len(all_rows)}, expected <= {max_rows}")

    # Validate total_value (sum of value columns)
    if "total_value" in expected:
        expected_total = expected["total_value"]
        actual_total: float = 0.0
        for row in all_rows:
            # Check for both "value" and "total_value" columns (handles GROUP BY queries)
            for col in ("value", "total_value"):
                if col in row:
                    with contextlib.suppress(ValueError, TypeError):
                        actual_total += float(row[col])
                    break  # Only count once per row
        # Allow 10% tolerance for rounding
        tolerance = expected_total * 0.1
        if abs(actual_total - expected_total) > tolerance:
            errors.append(f"total_value: got {actual_total}, expected {expected_total}")

    # Validate must_contain (single value must appear)
    if "must_contain" in expected:
        value = expected["must_contain"]
        if isinstance(value, list):
            for v in value:
                if str(v).lower() not in all_text:
                    errors.append(f"must_contain: '{v}' not found in results")
        elif str(value).lower() not in all_text:
            errors.append(f"must_contain: '{value}' not found in results")

    # Validate must_contain_all (all values must appear)
    if "must_contain_all" in expected:
        for value in expected["must_contain_all"]:
            if str(value).lower() not in all_text:
                errors.append(f"must_contain_all: '{value}' not found in results")

    # Validate must_contain_any (at least one value must appear)
    if "must_contain_any" in expected:
        values = expected["must_contain_any"]
        found_any = any(str(v).lower() in all_text for v in values)
        if not found_any:
            errors.append(f"must_contain_any: none of {values} found in results")

    # Validate must_not_contain (values must NOT appear - critical for filtering)
    if "must_not_contain" in expected:
        values = expected["must_not_contain"]
        if isinstance(values, list):
            for v in values:
                if str(v).lower() in all_text:
                    errors.append(f"must_not_contain: '{v}' found but should be excluded")
        elif str(values).lower() in all_text:
            errors.append(f"must_not_contain: '{values}' found but should be excluded")

    # Validate exact_values (specific column values that must match exactly)
    if "exact_values" in expected:
        for col, values in expected["exact_values"].items():
            actual_col_values = [row.get(col) for row in all_rows if col in row]
            # Convert to comparable format
            expected_set = set(str(v).lower() for v in values)
            actual_set = set(str(v).lower() for v in actual_col_values)
            if expected_set != actual_set:
                errors.append(f"exact_values[{col}]: got {actual_set}, expected {expected_set}")

    passed = len(errors) == 0
    return passed, errors


def get_paths_for_role(role: str | None = None) -> list[list[str]]:
    """
    Get all conversation paths for a specific role.

    Args:
        role: Filter by role - "sales", "csm", or "manager". If None, returns all paths.

    Returns:
        List of paths, where each path is a list of questions from starter to terminal.
    """
    starters, subgraph = _get_subgraph(role)
    max_depth = _compute_max_depth(starters)
    return _find_paths(starters, subgraph, max_depth)


def get_tree_stats(role: str | None = None) -> dict:
    """
    Get statistics about the question tree.

    Args:
        role: Filter by role - "sales", "csm", or "manager". If None, shows all.
    """
    starters, subgraph = _get_subgraph(role)
    max_depth = _compute_max_depth(starters)
    paths = _find_paths(starters, subgraph, max_depth)

    return {
        "role": role or "all",
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


def validate_tree(role: str | None = None) -> list[str]:
    """
    Validate the question tree for consistency.

    Args:
        role: Filter by role - "sales", "csm", or "manager". If None, validates all.

    Returns list of any issues found.
    """
    starters, subgraph = _get_subgraph(role)
    reachable = set(subgraph.nodes())
    issues = []

    # Check all starters are in tree
    for starter in starters:
        if starter not in _G:
            issues.append(f"Starter not in tree: {starter}")

    # Only check orphans when validating all roles
    if role is None:
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


def print_tree(role: str | None = None, max_depth: int | None = None) -> Tree:
    """
    Generate a Rich Tree representation of the question tree.

    Args:
        role: Filter by role - "sales", "csm", or "manager". If None, shows all.
        max_depth: Maximum depth to display (default: show all levels)

    Returns:
        Rich Tree object (print with rich.print or console.print).

    Usage:
        from rich import print
        print(print_tree())              # All trees
        print(print_tree("sales"))       # Sales Rep tree only
        print(print_tree("csm", max_depth=3))  # CSM tree, 3 levels deep
    """
    from rich.tree import Tree

    try:
        starters = _get_starters_for_role(role)
    except ValueError as e:
        # Return a tree with error message
        return Tree(f"[red]{e}[/red]")

    # Role labels
    role_labels = {
        "How's my pipeline?": "[bold cyan]SALES REP (jsmith)[/bold cyan]",
        "Any renewals at risk?": "[bold green]CSM (amartin)[/bold green]",
        "How's the team doing?": "[bold yellow]MANAGER[/bold yellow]",
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
    if len(starters) == 1:
        # Single role - use role label as root
        label = role_labels.get(starters[0], starters[0])
        root = Tree(label)
        starter_branch = root.add(f"[bold]{starters[0]}[/bold]")
        add_children(starter_branch, starters[0], 1)
    else:
        # All roles - create a root with children for each role
        root = Tree("[bold]Question Tree[/bold]")
        for starter in starters:
            label = role_labels.get(starter, starter)
            role_branch = root.add(label)
            starter_branch = role_branch.add(f"[bold]{starter}[/bold]")
            add_children(starter_branch, starter, 1)

    return root
