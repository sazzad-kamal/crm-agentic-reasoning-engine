"""
Tests for backend/agent/question_tree.py - hardcoded question tree for demos.
"""

from backend.agent.followup.tree import (
    get_starters,
    get_follow_ups,
    get_paths_for_role,
    get_tree_stats,
    validate_tree,
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
        """Returns expected number of starters (role-based)."""
        starters = get_starters()
        assert len(starters) == 3  # Sales Rep, CSM, Manager

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

    def test_contains_role_based_starters(self):
        """Contains role-based starter questions."""
        starters = get_starters()
        assert "How's my pipeline?" in starters  # Sales Rep
        assert "Any renewals at risk?" in starters  # CSM
        assert "How's the team doing?" in starters  # Manager


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

    def test_returns_empty_for_unknown_question(self):
        """Returns empty list for unknown question."""
        follow_ups = get_follow_ups("Unknown question that's not in the tree")
        assert follow_ups == []

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
# get_paths_for_role Tests
# =============================================================================

class TestGetPathsForRole:
    """Tests for get_paths_for_role function."""

    def test_returns_list(self):
        """Returns a list of paths."""
        paths = get_paths_for_role()
        assert isinstance(paths, list)

    def test_paths_are_lists(self):
        """Each path is a list of questions."""
        paths = get_paths_for_role()
        for path in paths:
            assert isinstance(path, list)

    def test_paths_contain_strings(self):
        """Each path contains string questions."""
        paths = get_paths_for_role()
        for path in paths:
            for question in path:
                assert isinstance(question, str)

    def test_all_paths_start_with_starter(self):
        """All paths start with a starter question."""
        paths = get_paths_for_role()
        starters = set(get_starters())
        for path in paths:
            assert path[0] in starters

    def test_generates_paths(self):
        """Generates multiple paths."""
        paths = get_paths_for_role()
        assert len(paths) > 0

    def test_role_filter_sales(self):
        """Filters paths by sales role."""
        paths = get_paths_for_role("sales")
        for path in paths:
            assert path[0] == "How's my pipeline?"

    def test_role_filter_csm(self):
        """Filters paths by csm role."""
        paths = get_paths_for_role("csm")
        for path in paths:
            assert path[0] == "Any renewals at risk?"

    def test_role_filter_manager(self):
        """Filters paths by manager role."""
        paths = get_paths_for_role("manager")
        for path in paths:
            assert path[0] == "How's the team doing?"


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
        assert stats["num_starters"] == len(get_starters())

    def test_has_num_questions(self):
        """Has num_questions key."""
        stats = get_tree_stats()
        assert "num_questions" in stats
        assert stats["num_questions"] > 0

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


