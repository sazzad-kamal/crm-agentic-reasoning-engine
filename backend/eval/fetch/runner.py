"""Fetch node evaluation - tests SQL planner in isolation."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb
import typer
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.eval.fetch.models import CaseResult, EvalResults, Question
from backend.eval.fetch.sql_judge import judge_sql_results
from backend.eval.shared.formatting import (
    build_eval_table,
    console,
    print_case,
    print_dim,
    print_error,
    print_failed_case,
    print_section_header,
    print_status,
    print_warning,
)

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
    if limit:
        questions = questions[:limit]

    results = EvalResults(total=len(questions))
    conn = get_connection()

    for i, question in enumerate(questions):
        if verbose:
            print_case(i + 1, len(questions), question.text, question.difficulty)

        case, row_count = _eval_question(question, conn)

        if case.passed:
            results.passed += 1

        if verbose:
            if case.errors:
                for err in case.errors:
                    print_warning(err, indent=4)
            else:
                print_status(case.passed, f"({row_count} rows, {case.latency_ms:.0f}ms)")

        results.cases.append(case)

    results.compute_aggregates()
    return results


def print_summary(results: EvalResults) -> None:
    """Print evaluation summary."""
    pass_rate_passed = results.pass_rate >= 0.85
    table = build_eval_table(
        title="Fetch Node Evaluation Summary",
        sections=[],
        aggregate_row=("Pass Rate", f"{results.pass_rate * 100:.1f}%", ">=85.0%", pass_rate_passed),
    )
    console.print(table)
    console.print(
        f"\nTotal: {results.total}, Passed: {results.passed}, Failed: {results.failed}, "
        f"Avg latency: {results.avg_latency_ms:.0f}ms"
    )

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        print_section_header(f"Failed Cases ({len(failed)})")

        # Show up to 10 failed cases
        for i, c in enumerate(failed[:10], 1):
            error = "; ".join(c.errors)
            print_failed_case(i, c.question.text, c.question.difficulty)
            print_error("Error", error, indent=3)
            if c.sql:
                sql_preview = c.sql[:200] + "..." if len(c.sql) > 200 else c.sql
                print_dim(f"SQL: {sql_preview}", indent=3)
            console.print()

        if len(failed) > 10:
            print_dim(f"... and {len(failed) - 10} more failures")


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run fetch node evaluation."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_sql_eval(limit=limit, verbose=verbose))


if __name__ == "__main__":
    typer.run(main)
