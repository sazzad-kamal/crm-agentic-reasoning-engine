"""
Hardcoded question tree for demo reliability.

This module provides a deterministic tree of questions and follow-ups,
ensuring 100% reliability for demos - no LLM generation variability.

Structure:
- 3 role-based starter questions:
  * Sales Rep (jsmith): "How's my pipeline?" - Pipeline focus
  * CSM (amartin): "Any renewals at risk?" - Retention focus
  * Manager (all): "How's the team doing?" - Aggregate view
- Each leads to 3 follow-ups (layer 2)
- Each leads to 3 follow-ups (layer 3)
- Each leads to 3 follow-ups (layer 4)
- Total: 81 paths (3 * 3 * 3 * 3)

Usage:
    from backend.agent.question_tree import get_follow_ups, get_starters, generate_all_paths

    # Get starter questions
    starters = get_starters()

    # Get follow-ups for a question
    follow_ups = get_follow_ups("How's my pipeline?")

    # Generate all paths for testing
    paths = generate_all_paths()

    # Visualize the tree as Mermaid
    print(to_mermaid())
"""

import json
from pathlib import Path

import networkx as nx

# =============================================================================
# Load JSON and Build Graph
# =============================================================================

_DATA_PATH = Path(__file__).parent / "data" / "question_tree.json"
with open(_DATA_PATH) as f:
    _raw_data = json.load(f)

# Extract metadata
_META = _raw_data.pop("_meta", {})
STARTERS: list[str] = _META.get("starters", [])

# Build directed graph
G = nx.DiGraph()

for question, node in _raw_data.items():
    G.add_node(question, company_id=node.get("company_id"))
    for follow_up in node["follow_ups"]:
        G.add_edge(question, follow_up)

# Backwards-compatible exports
STARTER_QUESTIONS = STARTERS
QUESTION_TREE = {q: {"company_id": G.nodes[q].get("company_id"), "follow_ups": list(G.successors(q))} for q in G.nodes()}
TERMINAL_FOLLOW_UPS: list[str] = []


# =============================================================================
# Public API
# =============================================================================


def get_starters() -> list[str]:
    """Get the starter questions."""
    return STARTERS.copy()


def get_follow_ups(question: str) -> list[str]:
    """
    Get follow-up questions for a given question.

    Returns hardcoded follow-ups from the tree, or empty list
    if the question isn't in the tree.
    """
    if question in G:
        return list(G.successors(question))
    return []


def get_company_id(question: str) -> str | None:
    """Get the company_id context for a question."""
    if question in G:
        return G.nodes[question].get("company_id")
    return None


def generate_all_paths(max_depth: int = 4) -> list[list[str]]:
    """
    Generate all possible conversation paths through the tree.

    Args:
        max_depth: Maximum number of questions in a path (default 4)

    Returns:
        List of paths, where each path is a list of questions.
        With 3 starters and 3 follow-ups at each level:
        - depth=2: 9 paths (3 * 3)
        - depth=3: 27 paths (3 * 3 * 3)
        - depth=4: 81 paths (3 * 3 * 3 * 3)
    """
    # Edge case: depth 1 means just the starters
    if max_depth == 1:
        return [[s] for s in STARTERS]

    # Find leaf nodes (questions with no follow-ups)
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]

    paths: list[list[str]] = []
    for starter in STARTERS:
        for leaf in leaves:
            try:
                for path in nx.all_simple_paths(G, starter, leaf, cutoff=max_depth - 1):
                    if len(path) <= max_depth:
                        paths.append(path)
            except nx.NetworkXNoPath:
                continue

    return paths


def get_tree_stats() -> dict:
    """Get statistics about the question tree."""
    paths = generate_all_paths()
    return {
        "num_starters": len(STARTERS),
        "num_questions": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_paths": len(paths),
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
    issues = []

    # Check all starters are in tree
    for starter in STARTERS:
        if starter not in G:
            issues.append(f"Starter not in tree: {starter}")

    # Check for orphaned questions (not reachable from starters)
    reachable: set[str] = set()
    for starter in STARTERS:
        reachable.add(starter)
        reachable |= nx.descendants(G, starter)

    for question in G.nodes():
        if question not in reachable:
            issues.append(f"Orphaned question (not reachable): {question}")

    # Check it's a valid DAG (no cycles)
    if not nx.is_directed_acyclic_graph(G):
        issues.append("Tree contains cycles!")

    return issues


def to_mermaid(max_label_length: int = 40) -> str:
    """
    Generate a Mermaid diagram of the question tree.

    Args:
        max_label_length: Truncate labels longer than this (default 40)

    Returns:
        Mermaid diagram string that can be rendered in markdown.

    Usage:
        print(to_mermaid())
        # Paste output into GitHub markdown or https://mermaid.live
    """
    lines = ["graph TD"]

    # Create node ID mapping (Mermaid doesn't like special chars in IDs)
    node_ids: dict[str, str] = {}
    for i, node in enumerate(G.nodes()):
        node_ids[node] = f"Q{i}"

    # Add edges with labels
    for src, dst in G.edges():
        src_id = node_ids[src]
        dst_id = node_ids[dst]

        # Truncate labels if too long
        src_label = src[:max_label_length] + "..." if len(src) > max_label_length else src
        dst_label = dst[:max_label_length] + "..." if len(dst) > max_label_length else dst

        # Escape quotes for Mermaid
        src_label = src_label.replace('"', "'")
        dst_label = dst_label.replace('"', "'")

        lines.append(f'    {src_id}["{src_label}"] --> {dst_id}["{dst_label}"]')

    # Style starter nodes
    for starter in STARTERS:
        if starter in node_ids:
            lines.append(f"    style {node_ids[starter]} fill:#e1f5fe")

    return "\n".join(lines)
