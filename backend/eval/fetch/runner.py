"""Fetch node evaluation - tests SQL planner in isolation."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.eval.fetch.models import CaseResult, EvalResults, Question
from backend.eval.fetch.sql_judge import judge_sql_results
from backend.eval.shared.formatting import build_eval_table, console

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


def run_sql_eval(
    limit: int | None = None,
    verbose: bool = False,
) -> EvalResults:
    """Run fetch node evaluation.

    For each question:
    1. Generate SQL via get_sql_plan()
    2. Execute SQL against DuckDB
    3. Validate results using LLM judge
    """
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = EvalResults(total=len(questions))
    conn = get_connection()

    for i, question in enumerate(questions):
        if verbose:
            console.print(
                f"\n[bold]Case {i + 1}/{len(questions)}:[/bold] {question.text} "
                f"[dim](d={question.difficulty})[/dim]"
            )

        start = time.time()
        sql = ""
        data: list[dict] = []
        passed = False
        errors: list[str] = []

        try:
            plan = get_sql_plan(question.text)
            sql = plan.sql
            data, sql_error = execute_sql(sql, conn)

            if sql_error:
                errors.append(f"SQL error: {sql_error}")
                if verbose:
                    console.print(f"  [red]SQL ERROR[/red]: {sql_error}")
            else:
                passed, errors = judge_sql_results(question.text, sql, {"query": data})
                if passed:
                    results.passed += 1

        except Exception as e:
            errors.append(f"Planner error: {e}")
            if verbose:
                console.print(f"  [red]PLANNER ERROR[/red]: {e}")

        latency = (time.time() - start) * 1000
        case = CaseResult(
            question=question,
            sql=sql,
            passed=passed,
            errors=errors,
            latency_ms=latency,
        )

        if verbose:
            if errors:
                for err in errors:
                    console.print(f"    [yellow]{err}[/yellow]")
            else:
                status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                console.print(f"  {status} ({len(data)} rows, {latency:.0f}ms)")

        results.cases.append(case)

    # Compute aggregate metrics
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
        console.print(f"\n[bold red]Failed Cases ({len(failed)})[/bold red]\n")

        # Show up to 10 failed cases
        for i, c in enumerate(failed[:10], 1):
            error = "; ".join(c.errors)
            console.print(f"[bold cyan]{i}. {c.question.text}[/bold cyan] [dim](d={c.question.difficulty})[/dim]")
            console.print(f"   [red]Error:[/red] {error}")
            if c.sql:
                console.print(f"   [dim]SQL:[/dim] {c.sql[:200]}...")
            console.print()

        if len(failed) > 10:
            console.print(f"[dim]... and {len(failed) - 10} more failures[/dim]")


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run fetch node evaluation."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_sql_eval(limit=limit, verbose=verbose))


if __name__ == "__main__":
    typer.run(main)
