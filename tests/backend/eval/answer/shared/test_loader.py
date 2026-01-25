"""Tests for backend.eval.answer.shared.loader module."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestLoadQuestions:
    """Tests for load_questions function."""

    def test_load_questions_all(self, monkeypatch, tmp_path):
        """Test loading all questions."""
        yaml_content = """
questions:
  - text: "Question 1"
    difficulty: 1
    expected_sql: "SELECT 1"
    expected_answer: "Answer 1"
  - text: "Question 2"
    difficulty: 2
    expected_sql: "SELECT 2"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.answer.shared.loader as loader_module

        monkeypatch.setattr(loader_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.answer.shared.loader import load_questions

        questions = load_questions()
        assert len(questions) == 2
        assert questions[0].text == "Question 1"
        assert questions[0].expected_sql == "SELECT 1"
        assert questions[0].expected_answer == "Answer 1"
        assert questions[1].expected_answer == ""  # Default


class TestGenerateAnswer:
    """Tests for generate_answer function."""

    def test_generate_answer_success(self, monkeypatch):
        """Test successful answer generation."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_answer
        from backend.eval.answer.shared.models import Question

        # Mock execute_sql
        monkeypatch.setattr(
            loader_module,
            "execute_sql",
            MagicMock(return_value=([{"value": 1}], None)),
        )

        # Mock call_answer_chain
        monkeypatch.setattr(
            loader_module,
            "call_answer_chain",
            MagicMock(return_value="The answer is 1."),
        )

        # Mock extract_suggested_action
        monkeypatch.setattr(
            loader_module,
            "extract_suggested_action",
            MagicMock(return_value=("The answer is 1.", None)),
        )

        q = Question(text="Test", difficulty=1, expected_sql="SELECT 1")
        mock_conn = MagicMock()

        answer, action, results, error = generate_answer(q, mock_conn)

        assert answer == "The answer is 1."
        assert action is None
        assert results == [{"value": 1}]
        assert error is None

    def test_generate_answer_sql_error(self, monkeypatch):
        """Test SQL error handling."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_answer
        from backend.eval.answer.shared.models import Question

        # Mock execute_sql to return error
        monkeypatch.setattr(
            loader_module,
            "execute_sql",
            MagicMock(return_value=([], "SQL syntax error")),
        )

        q = Question(text="Test", difficulty=1, expected_sql="INVALID SQL")
        mock_conn = MagicMock()

        answer, action, results, error = generate_answer(q, mock_conn)

        assert answer == ""
        assert action is None
        assert results == []
        assert "SQL error" in error

    def test_generate_answer_exception(self, monkeypatch):
        """Test exception handling."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_answer
        from backend.eval.answer.shared.models import Question

        # Mock execute_sql to raise exception
        monkeypatch.setattr(
            loader_module,
            "execute_sql",
            MagicMock(side_effect=Exception("Connection failed")),
        )

        q = Question(text="Test", difficulty=1, expected_sql="SELECT 1")
        mock_conn = MagicMock()

        answer, action, results, error = generate_answer(q, mock_conn)

        assert answer == ""
        assert action is None
        assert results == []
        assert "Error:" in error

    def test_generate_answer_with_action(self, monkeypatch):
        """Test answer generation with suggested action."""
        import backend.eval.answer.shared.loader as loader_module
        from backend.eval.answer.shared.loader import generate_answer
        from backend.eval.answer.shared.models import Question

        # Mock execute_sql
        monkeypatch.setattr(
            loader_module,
            "execute_sql",
            MagicMock(return_value=([{"status": "Active"}], None)),
        )

        # Mock call_answer_chain
        monkeypatch.setattr(
            loader_module,
            "call_answer_chain",
            MagicMock(return_value="Status is Active.\n\nSuggested action: Call customer"),
        )

        # Mock extract_suggested_action
        monkeypatch.setattr(
            loader_module,
            "extract_suggested_action",
            MagicMock(return_value=("Status is Active.", "Call customer")),
        )

        q = Question(text="What is the status?", difficulty=1, expected_sql="SELECT status FROM companies")
        mock_conn = MagicMock()

        answer, action, results, error = generate_answer(q, mock_conn)

        assert answer == "Status is Active."
        assert action == "Call customer"
        assert error is None
