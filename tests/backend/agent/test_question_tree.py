"""
Tests for backend/agent/followup/tree - hardcoded question tree for demos.
"""

from backend.agent.followup.tree import (
    get_starters,
    get_follow_ups,
)
from backend.eval.integration.tree import (
    get_all_paths,
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
        """Returns expected number of starters (3 CRM entities)."""
        starters = get_starters()
        assert len(starters) == 3

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

    def test_contains_entity_starters(self):
        """Contains starters for 3 CRM entities."""
        starters = get_starters()
        assert "What deals are in the pipeline?" in starters  # Opportunities
        assert "Which accounts are at risk?" in starters  # Companies
        assert "Which contacts haven't been contacted recently?" in starters  # Contacts


# =============================================================================
# get_follow_ups Tests
# =============================================================================

class TestGetFollowUps:
    """Tests for get_follow_ups function."""

    def test_returns_list(self):
        """Returns a list."""
        follow_ups = get_follow_ups("What deals are in the pipeline?")
        assert isinstance(follow_ups, list)

    def test_returns_follow_ups_for_known_question(self):
        """Returns follow-ups for a known question."""
        follow_ups = get_follow_ups("What deals are in the pipeline?")
        assert len(follow_ups) == 3
        assert "How are deals distributed by stage?" in follow_ups

    def test_returns_empty_for_unknown_question(self):
        """Returns empty list for unknown question."""
        follow_ups = get_follow_ups("Unknown question that's not in the tree")
        assert follow_ups == []

    def test_returns_copy(self):
        """Returns a copy, not the original list."""
        follow_ups1 = get_follow_ups("What deals are in the pipeline?")
        follow_ups2 = get_follow_ups("What deals are in the pipeline?")
        follow_ups1.append("modified")
        assert "modified" not in follow_ups2

    def test_depth_2_has_follow_ups(self):
        """Depth 2 questions have follow-ups."""
        follow_ups = get_follow_ups("How are deals distributed by stage?")
        assert len(follow_ups) == 3

    def test_companies_starter_has_follow_ups(self):
        """Companies starter has follow-ups."""
        follow_ups = get_follow_ups("Which accounts are at risk?")
        assert len(follow_ups) == 3
        assert "What's happening with Delta Health?" in follow_ups

    def test_contacts_starter_has_follow_ups(self):
        """Contacts starter has follow-ups."""
        follow_ups = get_follow_ups("Which contacts haven't been contacted recently?")
        assert len(follow_ups) == 3
        assert "Which contacts haven't been reached?" in follow_ups


# =============================================================================
# get_all_paths Tests
# =============================================================================

class TestGetAllPaths:
    """Tests for get_all_paths function."""

    def test_returns_list(self):
        """Returns a list of paths."""
        paths = get_all_paths()
        assert isinstance(paths, list)

    def test_paths_are_lists(self):
        """Each path is a list of questions."""
        paths = get_all_paths()
        for path in paths:
            assert isinstance(path, list)

    def test_paths_contain_strings(self):
        """Each path contains string questions."""
        paths = get_all_paths()
        for path in paths:
            for question in path:
                assert isinstance(question, str)

    def test_all_paths_start_with_starter(self):
        """All paths start with a starter question."""
        paths = get_all_paths()
        starters = set(get_starters())
        for path in paths:
            assert path[0] in starters

    def test_generates_paths(self):
        """Generates multiple paths."""
        paths = get_all_paths()
        assert len(paths) > 0

    def test_paths_cover_all_starters(self):
        """Paths cover all 3 starters."""
        paths = get_all_paths()
        starters_covered = {path[0] for path in paths}
        expected_starters = {
            "What deals are in the pipeline?",
            "Which accounts are at risk?",
            "Which contacts haven't been contacted recently?",
        }
        assert starters_covered == expected_starters


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
