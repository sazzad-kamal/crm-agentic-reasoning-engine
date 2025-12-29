"""
Tests for backend/agent/question_tree.py - hardcoded question tree for demos.
"""

import pytest

from backend.agent.question_tree import (
    get_starters,
    get_follow_ups,
    get_company_id,
    generate_all_paths,
    get_tree_stats,
    validate_tree,
    QUESTION_TREE,
    STARTER_QUESTIONS,
    TERMINAL_FOLLOW_UPS,
)


# =============================================================================
# get_starters Tests
# =============================================================================

class TestGetStarters:
    """Tests for get_starters function."""

    def test_returns_list(self):
        """Returns a list of starter questions."""
        starters = get_starters()
        assert isinstance(starters, list)

    def test_returns_expected_count(self):
        """Returns expected number of starters."""
        starters = get_starters()
        assert len(starters) == 5

    def test_starters_are_strings(self):
        """All starters are strings."""
        starters = get_starters()
        for starter in starters:
            assert isinstance(starter, str)
            assert len(starter) > 0

    def test_returns_copy(self):
        """Returns a copy, not the original list."""
        starters1 = get_starters()
        starters2 = get_starters()
        starters1.append("modified")
        assert "modified" not in starters2

    def test_contains_expected_starters(self):
        """Contains expected starter questions."""
        starters = get_starters()
        assert "What's going on with Acme Manufacturing?" in starters
        assert "Show me Beta Tech's pipeline" in starters
        assert "Which renewals are coming up this month?" in starters
        assert "What's our total pipeline?" in starters
        assert "Give me the full picture on Crown Foods" in starters


# =============================================================================
# get_follow_ups Tests
# =============================================================================

class TestGetFollowUps:
    """Tests for get_follow_ups function."""

    def test_returns_list(self):
        """Returns a list."""
        follow_ups = get_follow_ups("What's going on with Acme Manufacturing?")
        assert isinstance(follow_ups, list)

    def test_returns_follow_ups_for_known_question(self):
        """Returns follow-ups for a known question."""
        follow_ups = get_follow_ups("What's going on with Acme Manufacturing?")
        assert len(follow_ups) == 3
        assert "Show me Acme Manufacturing's opportunities" in follow_ups

    def test_returns_terminal_for_unknown_question(self):
        """Returns terminal follow-ups for unknown question."""
        follow_ups = get_follow_ups("Unknown question that's not in the tree")
        assert follow_ups == TERMINAL_FOLLOW_UPS

    def test_returns_copy(self):
        """Returns a copy, not the original list."""
        follow_ups1 = get_follow_ups("What's going on with Acme Manufacturing?")
        follow_ups2 = get_follow_ups("What's going on with Acme Manufacturing?")
        follow_ups1.append("modified")
        assert "modified" not in follow_ups2

    def test_layer_2_has_follow_ups(self):
        """Layer 2 questions have follow-ups."""
        follow_ups = get_follow_ups("Show me Acme Manufacturing's opportunities")
        assert len(follow_ups) == 3

    def test_layer_3_has_follow_ups(self):
        """Layer 3 questions have follow-ups."""
        follow_ups = get_follow_ups("What stage is the upgrade deal in?")
        assert len(follow_ups) == 3


# =============================================================================
# get_company_id Tests
# =============================================================================

class TestGetCompanyId:
    """Tests for get_company_id function."""

    def test_returns_company_id_for_acme_question(self):
        """Returns company ID for Acme question."""
        company_id = get_company_id("What's going on with Acme Manufacturing?")
        assert company_id == "ACME-MFG"

    def test_returns_company_id_for_beta_question(self):
        """Returns company ID for Beta Tech question."""
        company_id = get_company_id("Show me Beta Tech's pipeline")
        assert company_id == "BETA-TECH"

    def test_returns_none_for_general_question(self):
        """Returns None for general (non-company-specific) question."""
        company_id = get_company_id("Which renewals are coming up this month?")
        assert company_id is None

    def test_returns_none_for_unknown_question(self):
        """Returns None for unknown question."""
        company_id = get_company_id("Unknown question")
        assert company_id is None


# =============================================================================
# generate_all_paths Tests
# =============================================================================

class TestGenerateAllPaths:
    """Tests for generate_all_paths function."""

    def test_returns_list(self):
        """Returns a list of paths."""
        paths = generate_all_paths()
        assert isinstance(paths, list)

    def test_paths_are_lists(self):
        """Each path is a list of questions."""
        paths = generate_all_paths()
        for path in paths:
            assert isinstance(path, list)

    def test_paths_contain_strings(self):
        """Each path contains string questions."""
        paths = generate_all_paths()
        for path in paths:
            for question in path:
                assert isinstance(question, str)

    def test_all_paths_start_with_starter(self):
        """All paths start with a starter question."""
        paths = generate_all_paths()
        starters = set(STARTER_QUESTIONS)
        for path in paths:
            assert path[0] in starters

    def test_generates_expected_path_count(self):
        """Generates expected number of paths (3 starters * 3^3 = 81 if fully expanded)."""
        paths = generate_all_paths()
        # With 3 starters and varying depths, we should have multiple paths
        assert len(paths) > 0

    def test_default_max_depth(self):
        """Default max depth is 4."""
        paths = generate_all_paths()
        for path in paths:
            assert len(path) <= 4

    def test_custom_max_depth(self):
        """Respects custom max depth."""
        paths = generate_all_paths(max_depth=2)
        for path in paths:
            assert len(path) <= 2

    def test_depth_1_returns_starters(self):
        """Depth 1 returns just starter questions."""
        paths = generate_all_paths(max_depth=1)
        assert len(paths) == len(STARTER_QUESTIONS)
        for path in paths:
            assert len(path) == 1
            assert path[0] in STARTER_QUESTIONS


# =============================================================================
# get_tree_stats Tests
# =============================================================================

class TestGetTreeStats:
    """Tests for get_tree_stats function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        stats = get_tree_stats()
        assert isinstance(stats, dict)

    def test_has_num_starters(self):
        """Has num_starters key."""
        stats = get_tree_stats()
        assert "num_starters" in stats
        assert stats["num_starters"] == len(STARTER_QUESTIONS)

    def test_has_num_questions(self):
        """Has num_questions key."""
        stats = get_tree_stats()
        assert "num_questions" in stats
        assert stats["num_questions"] == len(QUESTION_TREE)

    def test_has_num_paths(self):
        """Has num_paths key."""
        stats = get_tree_stats()
        assert "num_paths" in stats
        assert stats["num_paths"] > 0

    def test_has_path_lengths(self):
        """Has path_lengths with min and max."""
        stats = get_tree_stats()
        assert "path_lengths" in stats
        assert "min" in stats["path_lengths"]
        assert "max" in stats["path_lengths"]


# =============================================================================
# validate_tree Tests
# =============================================================================

class TestValidateTree:
    """Tests for validate_tree function."""

    def test_returns_list(self):
        """Returns a list of issues."""
        issues = validate_tree()
        assert isinstance(issues, list)

    def test_tree_is_valid(self):
        """Current tree should be valid (no issues)."""
        issues = validate_tree()
        assert len(issues) == 0, f"Tree validation failed: {issues}"


# =============================================================================
# QUESTION_TREE Structure Tests
# =============================================================================

class TestQuestionTreeStructure:
    """Tests for QUESTION_TREE structure."""

    def test_all_starters_in_tree(self):
        """All starter questions are in the tree."""
        for starter in STARTER_QUESTIONS:
            assert starter in QUESTION_TREE

    def test_nodes_have_company_id(self):
        """All nodes have company_id field."""
        for question, node in QUESTION_TREE.items():
            assert "company_id" in node

    def test_nodes_have_follow_ups(self):
        """All nodes have follow_ups field."""
        for question, node in QUESTION_TREE.items():
            assert "follow_ups" in node
            assert isinstance(node["follow_ups"], list)

    def test_follow_ups_are_strings(self):
        """All follow-ups are strings."""
        for question, node in QUESTION_TREE.items():
            for follow_up in node["follow_ups"]:
                assert isinstance(follow_up, str)

    def test_company_ids_are_valid(self):
        """Company IDs are either None or valid strings."""
        # All 8 companies in the CRM plus None for global queries
        valid_company_ids = {
            "ACME-MFG", "BETA-TECH", "CROWN-FOODS", "DELTA-HEALTH",
            "EASTERN-TRAVEL", "FUSION-RETAIL", "GREEN-ENERGY", "HARBOR-LOGISTICS",
            None
        }
        for question, node in QUESTION_TREE.items():
            assert node["company_id"] in valid_company_ids


# =============================================================================
# TERMINAL_FOLLOW_UPS Tests
# =============================================================================

class TestTerminalFollowUps:
    """Tests for TERMINAL_FOLLOW_UPS constant."""

    def test_is_empty_list(self):
        """Terminal follow-ups is an empty list."""
        assert TERMINAL_FOLLOW_UPS == []

    def test_is_list(self):
        """Terminal follow-ups is a list."""
        assert isinstance(TERMINAL_FOLLOW_UPS, list)
