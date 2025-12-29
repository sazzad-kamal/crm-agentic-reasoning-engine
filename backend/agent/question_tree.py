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
"""

import json
from pathlib import Path
from typing import TypedDict


class QuestionNode(TypedDict):
    """A node in the question tree."""
    company_id: str | None  # Company context (if any)
    follow_ups: list[str]   # Next questions user can ask


# Load question tree from JSON
_DATA_PATH = Path(__file__).parent / "data" / "question_tree.json"
with open(_DATA_PATH) as f:
    _raw_data = json.load(f)

# Extract metadata and build tree
_META = _raw_data.pop("_meta", {})
QUESTION_TREE: dict[str, QuestionNode] = _raw_data
STARTER_QUESTIONS: list[str] = _META.get("starters", [])
TERMINAL_FOLLOW_UPS: list[str] = []


# =============================================================================
# Public API
# =============================================================================

def get_starters() -> list[str]:
    """Get the starter questions."""
    return STARTER_QUESTIONS.copy()


def get_follow_ups(question: str) -> list[str]:
    """
    Get follow-up questions for a given question.

    Returns hardcoded follow-ups from the tree, or terminal follow-ups
    if the question isn't in the tree.
    """
    node = QUESTION_TREE.get(question)
    if node:
        return node["follow_ups"].copy()
    return TERMINAL_FOLLOW_UPS.copy()


def get_company_id(question: str) -> str | None:
    """Get the company_id context for a question."""
    node = QUESTION_TREE.get(question)
    if node:
        return node.get("company_id")
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
    paths: list[list[str]] = []

    def _traverse(current_path: list[str], depth: int):
        if depth >= max_depth:
            paths.append(current_path.copy())
            return

        current_question = current_path[-1] if current_path else None

        if current_question is None:
            # Start with starters
            for starter in STARTER_QUESTIONS:
                _traverse([starter], depth + 1)
        else:
            # Get follow-ups for current question
            follow_ups = get_follow_ups(current_question)
            if not follow_ups or follow_ups == TERMINAL_FOLLOW_UPS:
                # Terminal node - end this path
                paths.append(current_path.copy())
                return

            for follow_up in follow_ups:
                _traverse(current_path + [follow_up], depth + 1)

    _traverse([], 0)
    return paths


def get_tree_stats() -> dict:
    """Get statistics about the question tree."""
    paths = generate_all_paths()
    return {
        "num_starters": len(STARTER_QUESTIONS),
        "num_questions": len(QUESTION_TREE),
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
    for starter in STARTER_QUESTIONS:
        if starter not in QUESTION_TREE:
            issues.append(f"Starter not in tree: {starter}")

    # Check for orphaned questions (not reachable from starters)
    reachable = set()

    def _mark_reachable(q: str):
        if q in reachable:
            return
        reachable.add(q)
        node = QUESTION_TREE.get(q)
        if node:
            for follow_up in node["follow_ups"]:
                _mark_reachable(follow_up)

    for starter in STARTER_QUESTIONS:
        _mark_reachable(starter)

    for question in QUESTION_TREE:
        if question not in reachable:
            issues.append(f"Orphaned question (not reachable): {question}")

    return issues
