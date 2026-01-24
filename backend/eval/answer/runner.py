"""Answer node evaluation runner."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import duckdb
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from backend.agent.answer.answerer import call_answer_chain, extract_suggested_action
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.eval.answer.judge import judge_suggested_action
from backend.eval.answer.models import CaseResult, EvalResults, Question
from backend.eval.shared.ragas import evaluate_single

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent.parent / "shared" / "questions.yaml"

console = Console()


def load_questions() -> list[Question]:
    """Load questions from shared YAML file."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)
    return [Question(**item) for item in data.get("questions", [])]


def _eval_case(question: Question, conn: duckdb.DuckDBPyConnection) -> CaseResult:
    """Evaluate single question: run expected SQL, generate answer, judge."""
    start = time.time()
    errors: list[str] = []

    try:
        # Step 1: Execute expected SQL to get deterministic results
        sql_results, sql_error = execute_sql(question.expected_sql, conn)
        if sql_error:
            errors.append(f"SQL error: {sql_error}")
            return CaseResult(
                question=question.text,
                answer="",
                suggested_action=None,
                latency_ms=0,
                errors=errors,
            )

        # Step 2: Generate answer
        raw_answer = call_answer_chain(question.text, sql_results={"rows": sql_results})
        answer, suggested_action = extract_suggested_action(raw_answer)

        # Step 3: RAGAS evaluation
        contexts = [json.dumps(sql_results, default=str)] if sql_results else []
        # Only pass reference_answer if we have one (enables answer_correctness metric)
        ref_answer = question.expected_answer if question.expected_answer else None
        ragas = evaluate_single(question.text, answer, contexts, reference_answer=ref_answer)

        # Step 4: Action judge (if action present)
        action_metrics = (False, 0.0, 0.0, 0.0, "")
        if suggested_action:
            action_metrics = judge_suggested_action(question.text, answer, suggested_action)

        latency = int((time.time() - start) * 1000)
        return CaseResult(
            question=question.text,
            answer=answer,
            suggested_action=suggested_action,
            latency_ms=latency,
            faithfulness_score=ragas["faithfulness"],  # type: ignore[arg-type]
            relevance_score=ragas["answer_relevancy"],  # type: ignore[arg-type]
            answer_correctness_score=ragas.get("answer_correctness", 0.0),  # type: ignore[arg-type]
            action_relevance=action_metrics[1],
            action_actionability=action_metrics[2],
            action_appropriateness=action_metrics[3],
            action_passed=action_metrics[0],
            errors=errors,
        )
    except Exception as e:
        logger.warning(f"Error evaluating '{question.text[:50]}': {e}")
        return CaseResult(
            question=question.text,
            answer="",
            suggested_action=None,
            latency_ms=int((time.time() - start) * 1000),
            errors=[f"Error: {e}"],
        )


def run_answer_eval(limit: int | None = None, verbose: bool = False) -> EvalResults:
    """Run answer node evaluation."""
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = EvalResults(total=len(questions))
    conn = get_connection()

    with Progress(
        TextColumn("[cyan]Evaluating answers"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("", total=len(questions))
        for q in questions:
            case = _eval_case(q, conn)
            results.cases.append(case)
            if case.passed:
                results.passed += 1
            if verbose:
                status = "[green]PASS[/green]" if case.passed else "[red]FAIL[/red]"
                console.print(f"  {status} {q.text[:60]}")
            progress.advance(task)

    results.compute_aggregates()
    return results


def print_summary(results: EvalResults) -> None:
    """Print evaluation summary."""
    passed = results.pass_rate >= 0.80
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"

    console.print()
    console.print("[bold]Answer Node Evaluation[/bold]")
    console.print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>=80.0% SLO) {status}")
    console.print(
        f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}, "
        f"Avg latency: {results.avg_latency_ms:.0f}ms"
    )
    console.print(
        f"RAGAS: F={results.avg_faithfulness:.2f} R={results.avg_relevance:.2f} "
        f"C={results.avg_answer_correctness:.2f} | Action: {results.action_pass_rate * 100:.0f}%"
    )

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        console.print(f"\n[bold]Failed Cases ({len(failed)})[/bold]\n")

        for i, c in enumerate(failed[:10], 1):
            error = "; ".join(c.errors) if c.errors else f"F={c.faithfulness_score:.2f} R={c.relevance_score:.2f}"
            console.print(f"{i}. {c.question[:60]}")
            console.print(f"   [dim]Scores: {error}[/dim]")
            if c.answer:
                console.print(f"   [dim]Answer: {c.answer[:100]}...[/dim]")
            console.print()

        if len(failed) > 10:
            console.print(f"[dim]... and {len(failed) - 10} more failures[/dim]")


__all__ = ["load_questions", "run_answer_eval", "print_summary"]
