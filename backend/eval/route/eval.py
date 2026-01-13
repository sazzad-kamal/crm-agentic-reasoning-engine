"""SQL Planner evaluation - tests generated SQL against expected results."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from backend.agent.datastore.connection import get_connection
from backend.agent.followup.tree import get_expected_sql_results, validate_sql_results
from backend.agent.route.sql_planner import get_sql_plan

console = Console()

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_test_questions() -> list[str]:
    """Load all questions from expected_sql_results.yaml (memory-independent)."""
    with open(_FIXTURES_DIR / "expected_sql_results.yaml") as f:
        data = yaml.safe_load(f)
    return list(data.keys())


@dataclass
class CaseResult:
    """Result for a single test case."""

    question: str
    sql: str
    passed: bool
    row_count: int = 0
    errors: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

    total: int = 0
    passed: int = 0
    sql_executed: int = 0
    sql_failed: int = 0
    cases: list[CaseResult] = field(default_factory=list)


def run_sql_eval(
    questions: list[str] | None = None,
    verbose: bool = False,
    limit: int | None = None,
) -> EvalResults:
    """
    Run SQL planner evaluation.

    For each question:
    1. Generate SQL via get_sql_plan()
    2. Execute SQL against DuckDB
    3. Validate results against expected_sql_results.yaml assertions

    Args:
        questions: List of questions to test (default: TEST_QUESTIONS)
        verbose: Print detailed output

    Returns:
        EvalResults with per-case and aggregate metrics
    """
    questions = questions or _load_test_questions()
    if limit:
        questions = questions[:limit]
    results = EvalResults(total=len(questions))
    conn = get_connection()

    for i, question in enumerate(questions):
        if verbose:
            console.print(f"\n[bold]Case {i + 1}/{len(questions)}:[/bold] {question}")

        try:
            # Get SQL from planner
            plan = get_sql_plan(question)
            sql = plan.sql

            if verbose:
                console.print(f"  SQL: {sql[:80]}...")

            # Execute SQL
            try:
                result = conn.execute(sql)
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                data = [dict(zip(columns, row, strict=True)) for row in rows]
                results.sql_executed += 1

                # Validate against expected results
                sql_results = {"query": data}
                passed, errors = validate_sql_results(question, sql_results)

                # If no assertions defined, check if we got results
                if get_expected_sql_results(question) is None:
                    passed = len(data) > 0
                    errors = [] if passed else ["No results returned"]

                if passed:
                    results.passed += 1

                case = CaseResult(
                    question=question,
                    sql=sql,
                    passed=passed,
                    row_count=len(data),
                    errors=errors,
                )

                if verbose:
                    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                    console.print(f"  {status} ({len(data)} rows)")
                    if errors:
                        for err in errors:
                            console.print(f"    [yellow]{err}[/yellow]")

            except Exception as e:
                results.sql_failed += 1
                case = CaseResult(
                    question=question,
                    sql=sql,
                    passed=False,
                    error=f"SQL error: {e}",
                )
                if verbose:
                    console.print(f"  [red]SQL ERROR[/red]: {e}")

        except Exception as e:
            case = CaseResult(
                question=question,
                sql="",
                passed=False,
                error=f"Planner error: {e}",
            )
            if verbose:
                console.print(f"  [red]PLANNER ERROR[/red]: {e}")

        results.cases.append(case)

    return results


def print_summary(results: EvalResults) -> None:
    """Print evaluation summary."""
    console.print("\n[bold]SQL Planner Evaluation Summary[/bold]")
    console.print(f"Total: {results.total}, Passed: {results.passed}, Failed: {results.total - results.passed}")
    console.print(f"Pass Rate: {results.passed / results.total * 100:.1f}%")
    console.print(f"SQL Executed: {results.sql_executed}, SQL Failed: {results.sql_failed}")

    # Failed cases table
    failed = [c for c in results.cases if not c.passed]
    if failed:
        table = Table(title=f"Failed Cases ({len(failed)})")
        table.add_column("Question", style="cyan", max_width=40)
        table.add_column("Error", style="red", max_width=50)

        for c in failed[:10]:
            error = c.error or "; ".join(c.errors[:2])
            table.add_row(c.question[:40], error[:50])

        console.print(table)


def main() -> None:
    """Run eval from command line."""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="SQL Planner Evaluation")
    parser.add_argument("--limit", type=int, help="Limit number of questions to test")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    console.print("[bold]SQL Planner Evaluation[/bold]\n")
    results = run_sql_eval(verbose=not args.quiet, limit=args.limit)
    print_summary(results)


if __name__ == "__main__":
    main()
