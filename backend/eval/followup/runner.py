"""Followup suggestion evaluation - tests follow-up generation quality."""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from backend.agent.followup.suggester import generate_follow_up_suggestions
from backend.eval.followup.judge import judge_followup_suggestions
from backend.eval.followup.models import CaseResult, EvalResults, Question
from backend.eval.shared import console
from backend.eval.shared.formatting import build_eval_table

# Path to questions file
QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"


def load_questions() -> list[Question]:
    """Load questions from YAML file."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)

    return [
        Question(text=item["text"], context=item.get("context", ""))
        for item in data.get("questions", [])
    ]


def run_followup_eval(
    questions: list[Question] | None = None,
    verbose: bool = False,
    limit: int | None = None,
    use_hardcoded_tree: bool = True,
) -> EvalResults:
    """
    Run followup suggestion evaluation.

    For each question:
    1. Generate follow-up suggestions via generate_follow_up_suggestions()
    2. Evaluate quality using LLM judge

    Args:
        questions: List of questions to test (default: from questions.yaml)
        verbose: Print detailed output
        limit: Max number of questions to test
        use_hardcoded_tree: Whether to allow hardcoded tree (default: True)

    Returns:
        EvalResults with per-case and aggregate metrics
    """
    if questions is None:
        questions = load_questions()

    if limit:
        questions = questions[:limit]

    results = EvalResults(total=len(questions))

    for i, question in enumerate(questions):
        if verbose:
            console.print(
                f"\n[bold]Case {i + 1}/{len(questions)}:[/bold] {question.text}"
            )
            if question.context:
                console.print(f"  [dim]({question.context})[/dim]")

        case_start = time.time()

        # Generate suggestions
        suggestions: list[str] = []
        errors: list[str] = []

        try:
            suggestions = generate_follow_up_suggestions(
                question=question.text,
                use_hardcoded_tree=use_hardcoded_tree,
            )
        except Exception as e:
            errors.append(f"Generation error: {e}")
            if verbose:
                console.print(f"  [red]GENERATION ERROR[/red]: {e}")

        # Judge quality
        passed = False
        relevance_score = 0.0
        diversity_score = 0.0

        if suggestions and not errors:
            try:
                passed, relevance_score, diversity_score, judge_errors = (
                    judge_followup_suggestions(question.text, suggestions)
                )
                errors.extend(judge_errors)
            except Exception as e:
                errors.append(f"Judge error: {e}")
                if verbose:
                    console.print(f"  [red]JUDGE ERROR[/red]: {e}")

        latency_ms = (time.time() - case_start) * 1000

        case = CaseResult(
            question=question.text,
            suggestions=suggestions,
            passed=passed,
            relevance_score=relevance_score,
            diversity_score=diversity_score,
            errors=errors,
            latency_ms=latency_ms,
        )

        if passed:
            results.passed += 1

        if verbose:
            if errors:
                for err in errors:
                    console.print(f"    [yellow]{err}[/yellow]")
            else:
                status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                console.print(
                    f"  {status} (rel={relevance_score:.2f}, div={diversity_score:.2f}, "
                    f"{latency_ms:.0f}ms)"
                )
                if suggestions:
                    for s in suggestions:
                        console.print(f"    [dim]- {s}[/dim]")

        results.cases.append(case)

    # Compute aggregate metrics
    results.compute_aggregates()

    return results


def print_summary(results: EvalResults) -> None:
    """Print evaluation summary with detailed metrics."""
    # Build sections for the table
    relevance_passed = results.avg_relevance >= 0.7
    diversity_passed = results.avg_diversity >= 0.5

    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]] = [
        (
            "Quality",
            [
                ("  Relevance", f"{results.avg_relevance * 100:.1f}%", ">=70.0%", relevance_passed),
                ("  Diversity", f"{results.avg_diversity * 100:.1f}%", ">=50.0%", diversity_passed),
            ],
        ),
        (
            "Latency",
            [
                ("  Avg total", f"{results.avg_latency_ms:.0f}ms", None, None),
            ],
        ),
    ]

    # Build and print table
    pass_rate_passed = results.pass_rate >= 0.8
    table = build_eval_table(
        title="Followup Suggestion Evaluation Summary",
        sections=sections,
        aggregate_row=("Pass Rate", f"{results.pass_rate * 100:.1f}%", ">=80.0%", pass_rate_passed),
    )
    console.print(table)

    # Stats
    console.print(f"\nTotal: {results.total}, Passed: {results.passed}, Failed: {results.failed}")

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        console.print(f"\n[bold red]Failed Cases ({len(failed)})[/bold red]\n")

        for i, c in enumerate(failed[:10], 1):
            error = "; ".join(c.errors) if c.errors else "Quality below threshold"
            console.print(f"[bold cyan]{i}. {c.question}[/bold cyan]")
            console.print(f"   [red]Error:[/red] {error}")
            console.print(f"   [dim]Relevance: {c.relevance_score:.2f}, Diversity: {c.diversity_score:.2f}[/dim]")
            if c.suggestions:
                console.print("   [dim]Suggestions:[/dim]")
                for s in c.suggestions:
                    console.print(f"     - {s}")
            console.print()

        if len(failed) > 10:
            console.print(f"[dim]... and {len(failed) - 10} more failures[/dim]")


__all__ = ["run_followup_eval", "print_summary", "load_questions"]
