"""Fetch node evaluation - tests SQL planner and RAG in isolation."""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.rag.search import search_entity_context
from backend.agent.fetch.sql.connection import get_connection
from backend.eval.fetch.models import CaseResult, EvalResults, Question
from backend.eval.fetch.sql_judge import judge_sql_results
from backend.eval.shared import console, evaluate_single, measure_latency_ms

# Path to questions file
QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"


def load_questions(
    difficulty_filter: list[int] | None = None,
    rag_only_filter: bool | None = None,
) -> list[Question]:
    """
    Load questions from YAML file.

    Args:
        difficulty_filter: Only include questions with these difficulty levels
        rag_only_filter: If True, only RAG questions; if False, only SQL; if None, all

    Returns:
        List of Question objects
    """
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)

    questions = []
    for item in data.get("questions", []):
        q = Question(
            text=item["text"],
            difficulty=item.get("difficulty", 1),
            rag_only=item.get("rag_only", False),
        )

        # Apply filters
        if difficulty_filter and q.difficulty not in difficulty_filter:
            continue
        if rag_only_filter is True and not q.rag_only:
            continue
        if rag_only_filter is False and q.rag_only:
            continue

        questions.append(q)

    return questions


def run_sql_eval(
    questions: list[Question] | None = None,
    verbose: bool = False,
    limit: int | None = None,
    difficulty_filter: list[int] | None = None,
) -> EvalResults:
    """
    Run fetch node evaluation.

    For each question:
    1. Generate SQL via get_sql_plan()
    2. Execute SQL against DuckDB
    3. Validate results using LLM judge

    Args:
        questions: List of questions to test (default: from questions.yaml)
        verbose: Print detailed output
        limit: Max number of questions to test
        difficulty_filter: Only test questions with these difficulty levels

    Returns:
        EvalResults with per-case and aggregate metrics
    """
    if questions is None:
        questions = load_questions(difficulty_filter=difficulty_filter, rag_only_filter=None)

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

        case_start = time.time()

        # Initialize case variables
        sql = ""
        sql_gen_latency = 0.0
        sql_exec_latency = 0.0
        rag_latency = 0.0
        rag_precision = None
        rag_recall = None
        data: list[dict] = []
        passed = False
        errors: list[str] = []
        error: str | None = None

        try:
            # Get SQL from planner
            sql_gen_start = time.time()
            plan = get_sql_plan(question.text)
            sql = plan.sql
            sql_gen_latency = measure_latency_ms(sql_gen_start)

            # Execute SQL
            try:
                sql_exec_start = time.time()
                result = conn.execute(sql)
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                data = [dict(zip(columns, row, strict=True)) for row in rows]
                sql_exec_latency = measure_latency_ms(sql_exec_start)
                results.sql_executed += 1

                # Validate using LLM judge
                sql_results = {"query": data}
                passed, errors = judge_sql_results(question.text, sql, sql_results)

                # Evaluate RAG if needed
                if plan.needs_rag:
                    rag_start = time.time()
                    try:
                        # Get entity IDs from data for RAG filtering
                        entity_ids = {}
                        if data:
                            row = data[0]
                            for key in ("company_id", "contact_id", "opportunity_id"):
                                if key in row and row[key]:
                                    entity_ids[key] = str(row[key])

                        if entity_ids:
                            context, _ = search_entity_context(question.text, entity_ids)
                            rag_latency = measure_latency_ms(rag_start)
                            results.rag_invoked += 1

                            # Evaluate RAG quality with RAGAS
                            if context:
                                chunks = context.split("\n\n---\n\n")
                                ragas_scores = evaluate_single(
                                    question=question.text,
                                    answer=context,
                                    contexts=chunks,
                                )
                                precision_val = ragas_scores.get("context_precision", 0.0)
                                recall_val = ragas_scores.get("context_recall", 0.0)
                                rag_precision = float(precision_val) if isinstance(precision_val, (int, float)) else 0.0
                                rag_recall = float(recall_val) if isinstance(recall_val, (int, float)) else 0.0
                    except Exception as e:
                        if verbose:
                            console.print(f"    [yellow]RAG warning: {e}[/yellow]")

                if passed:
                    results.passed += 1

            except Exception as e:
                results.sql_failed += 1
                error = f"SQL error: {e}"
                if verbose:
                    console.print(f"  [red]SQL ERROR[/red]: {e}")

        except Exception as e:
            error = f"Planner error: {e}"
            if verbose:
                console.print(f"  [red]PLANNER ERROR[/red]: {e}")

        total_latency = measure_latency_ms(case_start)
        case = CaseResult(
            question=question.text,
            difficulty=question.difficulty,
            rag_only=question.rag_only,
            sql=sql,
            passed=passed,
            row_count=len(data),
            errors=errors,
            error=error,
            sql_gen_latency_ms=sql_gen_latency,
            sql_exec_latency_ms=sql_exec_latency,
            rag_latency_ms=rag_latency,
            total_latency_ms=total_latency,
            rag_precision=rag_precision,
            rag_recall=rag_recall,
        )

        if verbose and not error:
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            rag_info = f", RAG={rag_latency:.0f}ms" if rag_latency > 0 else ""
            console.print(f"  {status} ({len(data)} rows, {total_latency:.0f}ms{rag_info})")
            if errors:
                for err in errors:
                    console.print(f"    [yellow]{err}[/yellow]")

        results.cases.append(case)

    # Compute aggregate metrics
    results.compute_aggregates()

    return results


def print_summary(results: EvalResults) -> None:
    """Print evaluation summary with detailed metrics."""
    from backend.eval.shared.formatting import build_eval_table

    # Build sections for the table
    sql_passed = results.sql_correctness >= 0.9
    sections = [
        (
            "SQL",
            [
                ("  Correctness", f"{results.sql_correctness * 100:.1f}%", ">=90.0%", sql_passed),
                ("  Gen latency", f"{results.avg_sql_gen_latency_ms:.0f}ms", None, None),
                ("  Exec latency", f"{results.avg_sql_exec_latency_ms:.0f}ms", None, None),
            ],
        ),
    ]

    # Add RAG section if applicable
    if results.rag_invoked > 0:
        rag_precision_passed = results.avg_rag_precision >= 0.8
        rag_recall_passed = results.avg_rag_recall >= 0.7
        sections.append(
            (
                "RAG",
                [
                    ("  Precision", f"{results.avg_rag_precision * 100:.1f}%", ">=80.0%", rag_precision_passed),
                    ("  Recall", f"{results.avg_rag_recall * 100:.1f}%", ">=70.0%", rag_recall_passed),
                    ("  Latency", f"{results.avg_rag_latency_ms:.0f}ms", None, None),
                ],
            )
        )

    # Add latency section
    sections.append(
        (
            "Latency",
            [
                ("  Avg total", f"{results.avg_total_latency_ms:.0f}ms", None, None),
            ],
        )
    )

    # Build and print table
    pass_rate_passed = results.pass_rate >= 0.85
    table = build_eval_table(
        title="Fetch Node Evaluation Summary",
        sections=sections,
        aggregate_row=("Pass Rate", f"{results.pass_rate * 100:.1f}%", ">=85.0%", pass_rate_passed),
    )
    console.print(table)

    # Stats
    console.print(
        f"\nTotal: {results.total}, Passed: {results.passed}, Failed: {results.failed}"
    )
    console.print(f"SQL Executed: {results.sql_executed}, SQL Failed: {results.sql_failed}")
    if results.rag_invoked > 0:
        console.print(f"RAG Invoked: {results.rag_invoked}")

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        console.print(f"\n[bold red]Failed Cases ({len(failed)})[/bold red]\n")

        # Show up to 10 failed cases
        for i, c in enumerate(failed[:10], 1):
            error = c.error or "; ".join(c.errors)
            console.print(f"[bold cyan]{i}. {c.question}[/bold cyan] [dim](d={c.difficulty})[/dim]")
            console.print(f"   [red]Error:[/red] {error}")
            if c.sql:
                console.print(f"   [dim]SQL:[/dim] {c.sql[:200]}...")
            console.print()

        if len(failed) > 10:
            console.print(f"[dim]... and {len(failed) - 10} more failures[/dim]")


__all__ = ["run_sql_eval", "print_summary", "load_questions"]
