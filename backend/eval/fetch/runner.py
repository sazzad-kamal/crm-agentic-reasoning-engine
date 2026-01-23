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
from backend.eval.fetch.sql_judge import judge_sql_results

# Path to questions file
QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"


def load_questions() -> list[Question]:
    """Load questions from YAML file."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)

    return [
        Question(text=item["text"], difficulty=item.get("difficulty", 1))
        for item in data.get("questions", [])
    ]


def _eval_question(
    question: Question,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[CaseResult, int]:
    """Evaluate a single question.

    Returns:
        Tuple of (CaseResult, row_count)
    """
    start = time.time()
    sql = ""
    row_count = 0
    passed = False
    errors: list[str] = []

    try:
        plan = get_sql_plan(question.text)
        sql = plan.sql
        data, sql_error = execute_sql(sql, conn)
        row_count = len(data)

        if sql_error:
            errors.append(f"SQL error: {sql_error}")
        else:
            passed, errors = judge_sql_results(question.text, sql, {"query": data})

    except Exception as e:
        errors.append(f"Planner error: {e}")

    latency = (time.time() - start) * 1000
    case = CaseResult(
        question=question,
        sql=sql,
        passed=passed,
        errors=errors,
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
