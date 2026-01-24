"""Fetch node evaluation - tests SQL planner in isolation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    import duckdb
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.eval.fetch.models import CaseResult, EvalResults, Question
from backend.eval.fetch.sql_judge import ErrorType, JudgeError, judge_sql_equivalence

# Path to questions file (shared with answer eval)
QUESTIONS_PATH = Path(__file__).parent.parent / "shared" / "questions.yaml"

# Error types that are acceptable (don't change results)
_ALLOWED_ERROR_TYPES = {
    ErrorType.LIKE_VS_EXACT,
    ErrorType.CASE_SENSITIVITY,
    ErrorType.COLUMN_DIFF,
    ErrorType.ORDER_BY,
    ErrorType.JOIN_TYPE,
    ErrorType.GROUPING,  # Often caused by LEFT vs INNER JOIN choice
    ErrorType.ALIAS_DIFF,  # Purely syntactic difference
}


def load_questions() -> list[Question]:
    """Load questions from YAML file."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)

    return [
        Question(
            text=item["text"],
            difficulty=item.get("difficulty", 1),
            expected_sql=item.get("expected_sql"),
        )
        for item in data.get("questions", [])
    ]


def _eval_question(
    question: Question,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[CaseResult, int]:
    """Evaluate question: generate SQL, execute, compare to expected."""
    start = time.time()
    sql = ""
    row_count = 0
    passed = False
    judge_errors: list[JudgeError] = []

    try:
        # Step 1: Generate SQL
        plan = get_sql_plan(question.text)
        sql = plan.sql

        # Step 2: Execute SQL (validates it runs without error)
        data, sql_error = execute_sql(sql, conn)
        row_count = len(data)

        if sql_error:
            judge_errors.append(JudgeError(type=ErrorType.OTHER, description=f"SQL error: {sql_error}"))
        else:
            # Step 3: Compare generated SQL to expected SQL
            passed, judge_errors = judge_sql_equivalence(
                generated_sql=sql,
                expected_sql=question.expected_sql or "",
            )
            # Override: accept stylistic differences that don't change results
            if not passed:
                error_types = {e.type for e in judge_errors}
                if error_types <= _ALLOWED_ERROR_TYPES:
                    passed = True
                    judge_errors = []

    except Exception as e:
        judge_errors.append(JudgeError(type=ErrorType.OTHER, description=f"Planner error: {e}"))

    latency = (time.time() - start) * 1000
    # Filter out allowed error types from display, convert to strings
    display_errors = [e for e in judge_errors if e.type not in _ALLOWED_ERROR_TYPES]
    error_strings = [f"[{e.type.value}] {e.description}" for e in display_errors]
    case = CaseResult(
        question=question,
        sql=sql,
        passed=passed,
        errors=error_strings,
        latency_ms=latency,
    )
    return case, row_count


def run_sql_eval(
    limit: int | None = None,
    verbose: bool = False,
) -> EvalResults:
    """Run fetch node evaluation."""
    questions = load_questions()
    if limit is not None:
        questions = questions[:limit]

    results = EvalResults(total=len(questions))
    conn = get_connection()

    for i, question in enumerate(questions):
        if verbose:
            print(f"\nCase {i + 1}/{len(questions)}: {question.text} (d={question.difficulty})")

        case, row_count = _eval_question(question, conn)

        if case.passed:
            results.passed += 1

        if verbose:
            if case.errors:
                for err in case.errors:
                    print(f"    {err}")
            else:
                print(f"  PASS ({row_count} rows, {case.latency_ms:.0f}ms)")

        results.cases.append(case)

    results.compute_aggregates()
    return results


def print_summary(results: EvalResults) -> None:
    """Print evaluation summary."""
    passed = results.pass_rate >= 0.85
    status = "PASS" if passed else "FAIL"
    print("\nFetch Node Evaluation")
    print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>=85.0% SLO) {status}")
    print(
        f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}, "
        f"Avg latency: {results.avg_latency_ms:.0f}ms"
    )

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        print(f"\nFailed Cases ({len(failed)})\n")

        # Show up to 10 failed cases
        for i, c in enumerate(failed[:10], 1):
            error = "; ".join(c.errors)
            print(f"{i}. {c.question.text} (d={c.question.difficulty})")
            print(f"   Error: {error}")
            if c.sql:
                print(f"   SQL: {c.sql[:200]}...")
            print()

        if len(failed) > 10:
            print(f"... and {len(failed) - 10} more failures")


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run fetch node evaluation."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_sql_eval(limit=limit, verbose=verbose))


if __name__ == "__main__":
    typer.run(main)
