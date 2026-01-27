"""Followup suggestion evaluation runner using LLM Judge."""

from __future__ import annotations

import logging

import typer
from dotenv import load_dotenv

load_dotenv()

from backend.agent.followup.suggester import generate_follow_up_suggestions
from backend.eval.answer.shared.loader import load_questions
from backend.eval.followup.judge import judge_followup_suggestions
from backend.eval.followup.models import (
    SLO_FOLLOWUP_PASS_RATE,
    FollowupCaseResult,
    FollowupEvalResults,
)

logger = logging.getLogger(__name__)


def run_followup_eval(
    limit: int | None = None,
    use_hardcoded_tree: bool = True,
) -> FollowupEvalResults:
    """Run followup suggestion evaluation using LLM Judge."""
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = FollowupEvalResults(total=len(questions))

    for idx, q in enumerate(questions, 1):
        suggestions: list[str] = []
        errors: list[str] = []

        try:
            suggestions = generate_follow_up_suggestions(
                question=q.text,
                use_hardcoded_tree=use_hardcoded_tree,
            )
        except Exception as e:
            errors.append(f"Generation error: {e}")

        passed = False
        rel = 0.0
        div = 0.0
        explanation = ""

        if suggestions and not errors:
            try:
                passed, rel, div, explanation = judge_followup_suggestions(
                    q.text, suggestions
                )
            except Exception as e:
                logger.warning(f"Judge evaluation failed: {e}")
                errors.append(f"Judge failed: {e}")

        case = FollowupCaseResult(
            question=q.text,
            suggestions=suggestions,
            passed=passed,
            relevance=rel,
            diversity=div,
            explanation=explanation,
            errors=errors,
        )

        results.cases.append(case)
        status = "PASS" if case.passed else "FAIL"
        print(f"  [{idx}/{results.total}] {status} {q.text[:50]}")

    results.compute_aggregates()
    return results


def print_summary(results: FollowupEvalResults) -> None:
    """Print followup evaluation summary."""
    passed = results.pass_rate >= SLO_FOLLOWUP_PASS_RATE
    status = "PASS" if passed else "FAIL"

    print("\nFollowup Suggestion Evaluation (LLM Judge)")
    print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>={SLO_FOLLOWUP_PASS_RATE * 100:.1f}% SLO) {status}")
    print(f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}")
    print(f"  Followup Metrics: rel={results.avg_relevance:.2f} div={results.avg_diversity:.2f}")

    # Error cases
    error_cases = [c for c in results.cases if c.errors]
    if error_cases:
        print(f"\nError Cases ({len(error_cases)})\n")
        for i, c in enumerate(error_cases, 1):
            print(f"{i}. {c.question[:60]}")
            print(f"   Error: {'; '.join(c.errors)}")
            print()

    # Failed cases (non-error)
    failed = [c for c in results.cases if not c.passed and not c.errors]
    if failed:
        print(f"\nFailed Cases ({len(failed)})\n")
        for i, c in enumerate(failed, 1):
            print(f"{i}. {c.question}")
            print(f"   Scores: rel={c.relevance:.2f} div={c.diversity:.2f}")
            if c.explanation:
                print(f"   Judge: {c.explanation}")
            if c.suggestions:
                for s in c.suggestions:
                    print(f"   - {s}")
            print()


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    no_tree: bool = typer.Option(False, "--no-tree", help="Disable hardcoded tree, use LLM only"),
) -> None:
    """Run followup suggestion evaluation using LLM Judge."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_followup_eval(limit=limit, use_hardcoded_tree=not no_tree))


if __name__ == "__main__":
    typer.run(main)
