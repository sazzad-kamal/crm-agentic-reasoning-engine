"""Tests for backend.eval.shared.models module."""

from __future__ import annotations

from backend.eval.shared.models import BaseEvalResults


class TestBaseEvalResults:
    """Tests for BaseEvalResults base class."""

    def test_defaults(self):
        """Test default field values."""
        results = BaseEvalResults()
        assert results.total == 0
        assert results.passed == 0

    def test_failed_property(self):
        """Test failed property computes correctly."""
        results = BaseEvalResults(total=10, passed=7)
        assert results.failed == 3

    def test_pass_rate(self):
        """Test pass_rate property computes correctly."""
        results = BaseEvalResults(total=10, passed=8)
        assert results.pass_rate == 0.8

    def test_pass_rate_zero_total(self):
        """Test pass_rate returns 0.0 when total is zero."""
        results = BaseEvalResults(total=0, passed=0)
        assert results.pass_rate == 0.0

    def test_pass_rate_all_passed(self):
        """Test pass_rate returns 1.0 when all pass."""
        results = BaseEvalResults(total=5, passed=5)
        assert results.pass_rate == 1.0

    def test_pass_rate_none_passed(self):
        """Test pass_rate returns 0.0 when none pass."""
        results = BaseEvalResults(total=5, passed=0)
        assert results.pass_rate == 0.0
