"""Tests for backend.eval.act.multi_db_runner module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.eval.act.multi_db_runner import (
    DatabaseResult,
    MultiDbEvalSummary,
    _print_summary,
    run_multi_db_eval,
    save_csv_results,
    save_json_results,
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


class TestRunMultiDbEval:
    """Tests for run_multi_db_eval function."""

    @patch("backend.eval.act.multi_db_runner.evaluate_question")
    @patch("backend.eval.act.multi_db_runner.clear_api_cache")
    @patch("backend.eval.act.multi_db_runner.set_database")
    @patch("backend.eval.act.multi_db_runner.get_database")
    def test_run_multi_db_eval_basic(
        self, mock_get_db: MagicMock, mock_set_db: MagicMock, mock_clear_cache: MagicMock, mock_eval: MagicMock
    ) -> None:
        """Test basic multi-db evaluation run."""
        mock_get_db.return_value = "OriginalDB"
        mock_eval.return_value = QuestionResult(
            question="Q1",
            passed=True,
            faithfulness=0.8,
            relevancy=0.9,
            action_passed=True,
            total_latency_ms=1000,
        )

        summary = run_multi_db_eval(
            databases=["DB1", "DB2"],
            questions=["Q1"],
        )

        assert len(summary.databases) == 2
        assert summary.total_evaluations == 2
        assert mock_set_db.call_count == 3  # 2 DBs + restore original
        assert mock_clear_cache.call_count == 3
        assert mock_eval.call_count == 2

    @patch("backend.eval.act.multi_db_runner.evaluate_question")
    @patch("backend.eval.act.multi_db_runner.clear_api_cache")
    @patch("backend.eval.act.multi_db_runner.set_database")
    @patch("backend.eval.act.multi_db_runner.get_database")
    def test_run_multi_db_eval_with_db_failure(
        self, mock_get_db: MagicMock, mock_set_db: MagicMock, mock_clear_cache: MagicMock, mock_eval: MagicMock
    ) -> None:
        """Test evaluation continues when a database fails."""
        mock_get_db.return_value = "OriginalDB"
        mock_set_db.side_effect = [Exception("Auth failed"), None, None]
        mock_eval.return_value = QuestionResult(
            question="Q1",
            passed=True,
            faithfulness=0.8,
            relevancy=0.9,
            action_passed=True,
            total_latency_ms=1000,
        )

        summary = run_multi_db_eval(
            databases=["FailDB", "GoodDB"],
            questions=["Q1"],
        )

        assert len(summary.databases) == 2
        assert summary.databases[0].connection_failed is True
        assert summary.databases[1].connection_failed is False
        assert "FailDB" in summary.databases_failed

    @patch("backend.eval.act.multi_db_runner.evaluate_question")
    @patch("backend.eval.act.multi_db_runner.clear_api_cache")
    @patch("backend.eval.act.multi_db_runner.set_database")
    @patch("backend.eval.act.multi_db_runner.get_database")
    def test_run_multi_db_eval_uses_defaults(
        self, mock_get_db: MagicMock, mock_set_db: MagicMock, mock_clear_cache: MagicMock, mock_eval: MagicMock
    ) -> None:
        """Test that defaults are used when no args provided."""
        mock_get_db.return_value = "OriginalDB"
        mock_eval.return_value = QuestionResult(
            question="Q1",
            passed=True,
            faithfulness=0.8,
            relevancy=0.9,
            action_passed=True,
            total_latency_ms=1000,
        )

        summary = run_multi_db_eval()

        # Should use AVAILABLE_DATABASES (2) and DEMO_STARTERS (5)
        assert len(summary.databases) == 2
        assert summary.total_evaluations == 10  # 2 DBs x 5 questions


class TestPrintSummary:
    """Tests for _print_summary function."""

    def test_print_summary_outputs_to_stdout(self, capsys: object) -> None:
        """Test that print_summary outputs to stdout."""
        summary = MultiDbEvalSummary()
        db1 = DatabaseResult(database="DB1")
        db1.questions = [
            QuestionResult(question="Q1", passed=True, faithfulness=0.8, relevancy=0.9, action_passed=True, total_latency_ms=10000),
        ]
        db1.compute_aggregates()
        summary.databases = [db1]
        summary.compute_aggregates(["Q1"])

        _print_summary(summary, ["DB1"], ["Q1"])

        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert "OVERALL SUMMARY" in captured.out
        assert "DB1" in captured.out
        assert "Q1" in captured.out

    def test_print_summary_with_failed_database(self, capsys: object) -> None:
        """Test print_summary shows failed databases."""
        summary = MultiDbEvalSummary()
        db_failed = DatabaseResult(database="FailedDB", connection_failed=True, connection_error="Auth error")
        summary.databases = [db_failed]
        summary.databases_failed = ["FailedDB"]

        _print_summary(summary, ["FailedDB"], ["Q1"])

        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert "FailedDB" in captured.out
        assert "FAILED" in captured.out


class TestSaveJsonResults:
    """Tests for save_json_results function."""

    def test_save_json_results_creates_file(self) -> None:
        """Test that JSON file is created with correct structure."""
        summary = MultiDbEvalSummary()
        db1 = DatabaseResult(database="DB1")
        db1.questions = [
            QuestionResult(question="Q1", passed=True, faithfulness=0.8, relevancy=0.9, action_passed=True, total_latency_ms=10000),
        ]
        db1.compute_aggregates()
        summary.databases = [db1]
        summary.compute_aggregates(["Q1"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_json_results(summary, path)
            assert path.exists()

            import json
            with open(path) as f:
                data = json.load(f)

            assert "summary" in data
            assert "databases" in data
            assert len(data["databases"]) == 1
            assert data["databases"][0]["database"] == "DB1"
        finally:
            path.unlink()

    def test_save_json_results_includes_questions(self) -> None:
        """Test that JSON includes question details."""
        summary = MultiDbEvalSummary()
        db1 = DatabaseResult(database="DB1")
        db1.questions = [
            QuestionResult(
                question="Test Question",
                passed=True,
                faithfulness=0.85,
                relevancy=0.9,
                action_passed=True,
                total_latency_ms=5000,
                fetch_rows=10,
            ),
        ]
        db1.compute_aggregates()
        summary.databases = [db1]
        summary.compute_aggregates(["Test Question"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_json_results(summary, path)

            import json
            with open(path) as f:
                data = json.load(f)

            questions = data["databases"][0]["questions"]
            assert len(questions) == 1
            assert questions[0]["question"] == "Test Question"
            assert questions[0]["faithfulness"] == 0.85
            assert questions[0]["fetch_rows"] == 10
        finally:
            path.unlink()


class TestSaveCsvResults:
    """Tests for save_csv_results function."""

    def test_save_csv_results_creates_file(self) -> None:
        """Test that CSV file is created with headers and data."""
        summary = MultiDbEvalSummary()
        db1 = DatabaseResult(database="DB1")
        db1.questions = [
            QuestionResult(question="Q1", passed=True, faithfulness=0.8, relevancy=0.9, action_passed=True, total_latency_ms=10000),
        ]
        db1.compute_aggregates()
        summary.databases = [db1]
        summary.compute_aggregates(["Q1"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            save_csv_results(summary, path)
            assert path.exists()

            with open(path) as f:
                content = f.read()

            assert "database" in content
            assert "question" in content
            assert "DB1" in content
            assert "Q1" in content
        finally:
            path.unlink()

    def test_save_csv_results_handles_failed_database(self) -> None:
        """Test that CSV includes failed database row."""
        summary = MultiDbEvalSummary()
        db_failed = DatabaseResult(database="FailedDB", connection_failed=True, connection_error="Auth error")
        summary.databases = [db_failed]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            save_csv_results(summary, path)

            with open(path) as f:
                content = f.read()

            assert "FailedDB" in content
            assert "Auth error" in content
        finally:
            path.unlink()
