"""Shared loader utilities for answer evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import duckdb

from backend.agent.action.suggester import call_action_chain
from backend.agent.answer.answerer import call_answer_chain
from backend.agent.sql.executor import execute_sql
from backend.eval.answer.shared.models import Question

QUESTIONS_PATH = Path(__file__).parent.parent.parent / "shared" / "questions.yaml"


def load_questions() -> list[Question]:
    """Load questions from shared YAML file."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)
    return [Question(**item) for item in data.get("questions", [])]


def generate_answer(
    question: Question, conn: duckdb.DuckDBPyConnection
) -> tuple[str, list[dict], str | None]:
    """Execute SQL and generate answer.

    Returns:
        tuple: (answer_text, sql_results, error)
    """
    try:
        # Step 1: Execute expected SQL
        sql_results, sql_error = execute_sql(question.expected_sql, conn)
        if sql_error:
            return "", [], f"SQL error: {sql_error}"

        # Step 2: Call answer chain
        answer = call_answer_chain(question.text, sql_results={"rows": sql_results})

        return answer, sql_results, None

    except Exception as e:
        return "", [], f"Error: {e}"


def generate_action(question: str, answer: str) -> tuple[str | None, str | None]:
    """Generate suggested action. Returns (action, error)."""
    try:
        action = call_action_chain(question=question, answer=answer)
        return action, None
    except Exception as e:
        return None, f"Action error: {e}"


__all__ = ["generate_action", "generate_answer", "load_questions"]
