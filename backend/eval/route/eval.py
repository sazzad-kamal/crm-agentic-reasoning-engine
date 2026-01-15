"""SQL Planner evaluation - tests generated SQL using LLM judge."""

from dataclasses import dataclass, field

from rich.console import Console

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.eval.route.sql_judge import judge_sql_results
from backend.eval.tree import get_all_paths


def _get_unique_questions() -> list[str]:
    """Get all unique questions from the question tree."""
    paths = get_all_paths()
    seen = set()
    questions = []
    for path in paths:
        for q in path:
            if q not in seen:
                seen.add(q)
                questions.append(q)
    return questions

console = Console()


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

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


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
    3. Validate results using LLM judge

    Args:
        questions: List of questions to test (default: from eval tree)
        verbose: Print detailed output
        limit: Max number of questions to test

    Returns:
        EvalResults with per-case and aggregate metrics
    """
    questions = questions or _get_unique_questions()
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

            # Execute SQL
            try:
                result = conn.execute(sql)
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                data = [dict(zip(columns, row, strict=True)) for row in rows]
                results.sql_executed += 1

                # Validate using LLM judge
                sql_results = {"query": data}
                passed, errors = judge_sql_results(question, sql, sql_results)

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

    # Failed cases with SQL
    failed = [c for c in results.cases if not c.passed]
    if failed:
        console.print(f"\n[bold red]Failed Cases ({len(failed)})[/bold red]\n")
        for i, c in enumerate(failed, 1):
            error = c.error or "; ".join(c.errors)
            console.print(f"[bold cyan]{i}. {c.question}[/bold cyan]")
            console.print(f"   [red]Error:[/red] {error}")
            if c.sql:
                console.print(f"   [dim]SQL:[/dim] {c.sql}")
            console.print()


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
