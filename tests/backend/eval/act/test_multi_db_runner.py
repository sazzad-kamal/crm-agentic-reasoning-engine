"""Tests for backend.eval.act.multi_db_runner module."""

from __future__ import annotations

from backend.eval.act.multi_db_runner import (
    DatabaseResult,
    MultiDbEvalSummary,
)
from backend.eval.act.runner import QuestionResult


class TestDatabaseResult:
    """Tests for DatabaseResult dataclass."""

    def test_database_result_defaults(self):
        """Test DatabaseResult default values."""
        result = DatabaseResult(database="TestDB")
        assert result.database == "TestDB"
        assert result.questions == []
        assert result.pass_count == 0
        assert result.avg_faithfulness == 0.0
        assert result.avg_relevancy == 0.0
        assert result.avg_latency_ms == 0.0
        assert result.action_pass_rate == 0.0
        assert result.total_time_ms == 0
        assert result.connection_failed is False
        assert result.connection_error is None

    def test_database_result_with_connection_error(self):
        """Test DatabaseResult with connection failure."""
        result = DatabaseResult(
            database="FailedDB",
            connection_failed=True,
            connection_error="Auth failed",
        )
        assert result.connection_failed is True
        assert result.connection_error == "Auth failed"

    def test_compute_aggregates_empty(self):
        """Test compute_aggregates with no questions."""
        result = DatabaseResult(database="TestDB")
        result.compute_aggregates()
        assert result.pass_count == 0
        assert result.avg_faithfulness == 0.0

    def test_compute_aggregates_with_questions(self):
        """Test compute_aggregates calculates correct values."""
        result = DatabaseResult(database="TestDB")
        result.questions = [
            QuestionResult(
                question="Q1",
                passed=True,
                faithfulness=0.8,
                relevancy=0.9,
                total_latency_ms=1000,
                action_passed=True,
            ),
            QuestionResult(
                question="Q2",
                passed=True,
                faithfulness=0.7,
                relevancy=0.8,
                total_latency_ms=2000,
                action_passed=True,
            ),
            QuestionResult(
                question="Q3",
                passed=False,
                faithfulness=0.6,
                relevancy=0.7,
                total_latency_ms=3000,
                action_passed=False,
            ),
        ]
        result.compute_aggregates()

        assert result.pass_count == 2
        assert abs(result.avg_faithfulness - 0.7) < 0.001  # (0.8 + 0.7 + 0.6) / 3
        assert abs(result.avg_relevancy - 0.8) < 0.001  # (0.9 + 0.8 + 0.7) / 3
        assert result.avg_latency_ms == 2000  # (1000 + 2000 + 3000) / 3
        assert abs(result.action_pass_rate - 0.6667) < 0.01  # 2/3

    def test_compute_aggregates_skips_zero_scores(self):
        """Test that zero faithfulness/relevancy scores are excluded from averages."""
        result = DatabaseResult(database="TestDB")
        result.questions = [
            QuestionResult(
                question="Q1",
                passed=True,
                faithfulness=0.8,
                relevancy=0.9,
                total_latency_ms=1000,
                action_passed=True,
            ),
            QuestionResult(
                question="Q2",
                passed=False,
                faithfulness=0.0,  # Skip this in average
                relevancy=0.0,  # Skip this in average
                total_latency_ms=2000,
                action_passed=False,
            ),
        ]
        result.compute_aggregates()

        # Only Q1's scores counted for faithfulness/relevancy
        assert result.avg_faithfulness == 0.8
        assert result.avg_relevancy == 0.9
        # Latency includes all questions
        assert result.avg_latency_ms == 1500


class TestMultiDbEvalSummary:
    """Tests for MultiDbEvalSummary dataclass."""

    def test_multi_db_eval_summary_defaults(self):
        """Test MultiDbEvalSummary default values."""
        summary = MultiDbEvalSummary()
        assert summary.databases == []
        assert summary.total_evaluations == 0
        assert summary.total_passed == 0
        assert summary.overall_pass_rate == 0.0
        assert summary.databases_failed == []

    def test_compute_aggregates_empty(self):
        """Test compute_aggregates with no databases."""
        summary = MultiDbEvalSummary()
        summary.compute_aggregates(["Q1", "Q2"])
        assert summary.total_evaluations == 0
        assert summary.overall_pass_rate == 0.0

    def test_compute_aggregates_with_databases(self):
        """Test compute_aggregates with multiple databases."""
        summary = MultiDbEvalSummary()

        # DB1: 2 questions, both pass
        db1 = DatabaseResult(database="DB1")
        db1.questions = [
            QuestionResult(question="Q1", passed=True, faithfulness=0.8, relevancy=0.9, action_passed=True, total_latency_ms=1000),
            QuestionResult(question="Q2", passed=True, faithfulness=0.7, relevancy=0.8, action_passed=True, total_latency_ms=2000),
        ]
        db1.compute_aggregates()

        # DB2: 2 questions, one passes
        db2 = DatabaseResult(database="DB2")
        db2.questions = [
            QuestionResult(question="Q1", passed=True, faithfulness=0.9, relevancy=0.85, action_passed=True, total_latency_ms=1500),
            QuestionResult(question="Q2", passed=False, faithfulness=0.5, relevancy=0.6, action_passed=False, total_latency_ms=2500),
        ]
        db2.compute_aggregates()

        summary.databases = [db1, db2]
        summary.compute_aggregates(["Q1", "Q2"])

        assert summary.total_evaluations == 4
        assert summary.total_passed == 3
        assert summary.overall_pass_rate == 0.75  # 3/4
        assert len(summary.databases_failed) == 0

        # Per-question pass rates
        assert summary.question_pass_rates["Q1"] == 1.0  # 2/2 passed
        assert summary.question_pass_rates["Q2"] == 0.5  # 1/2 passed

    def test_compute_aggregates_with_failed_database(self):
        """Test compute_aggregates tracks failed databases."""
        summary = MultiDbEvalSummary()

        # DB1: successful
        db1 = DatabaseResult(database="DB1")
        db1.questions = [
            QuestionResult(question="Q1", passed=True, faithfulness=0.8, relevancy=0.9, action_passed=True, total_latency_ms=1000),
        ]
        db1.compute_aggregates()

        # DB2: connection failed
        db2 = DatabaseResult(database="DB2", connection_failed=True, connection_error="Auth failed")

        summary.databases = [db1, db2]
        summary.compute_aggregates(["Q1"])

        assert summary.total_evaluations == 1  # Only DB1's questions counted
        assert summary.databases_failed == ["DB2"]

    def test_compute_aggregates_metrics(self):
        """Test overall metrics are computed correctly."""
        summary = MultiDbEvalSummary()

        db1 = DatabaseResult(database="DB1")
        db1.questions = [
            QuestionResult(question="Q1", passed=True, faithfulness=0.8, relevancy=0.9, action_passed=True, total_latency_ms=10000),
            QuestionResult(question="Q2", passed=True, faithfulness=0.9, relevancy=0.85, action_passed=True, total_latency_ms=20000),
        ]
        db1.compute_aggregates()

        summary.databases = [db1]
        summary.compute_aggregates(["Q1", "Q2"])

        assert abs(summary.overall_avg_faithfulness - 0.85) < 0.001  # (0.8 + 0.9) / 2
        assert summary.overall_avg_relevancy == 0.875  # (0.9 + 0.85) / 2
        assert summary.overall_avg_latency_ms == 15000  # (10000 + 20000) / 2
        assert summary.overall_action_pass_rate == 1.0  # 2/2
