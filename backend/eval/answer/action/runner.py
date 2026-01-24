"""Action quality evaluation runner using LLM Judge."""

from __future__ import annotations

import logging

import typer

from backend.agent.fetch.sql.connection import get_connection
from backend.eval.answer.action.judge import judge_suggested_action
from backend.eval.answer.action.models import ActionCaseResult, ActionEvalResults
from backend.eval.answer.shared import generate_answer, load_questions

logger = logging.getLogger(__name__)


def run_action_eval(limit: int | None = None, verbose: bool = False) -> ActionEvalResults:
    """Run action quality evaluation using LLM Judge."""
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = ActionEvalResults(total=len(questions))
    conn = get_connection()

    for q in questions:
        answer, suggested_action, _, latency_ms, error = generate_answer(q, conn)

        if error:
            case = ActionCaseResult(
                question=q.text,
                answer="",
                suggested_action=None,
                latency_ms=latency_ms,
                errors=[error],
            )
        elif not suggested_action:
            # No action suggested - passes by default
            case = ActionCaseResult(
                question=q.text,
                answer=answer,
                suggested_action=None,
                latency_ms=latency_ms,
                action_passed=True,
            )
        else:
            # Judge the suggested action
            judge_passed, rel, act, app, _ = judge_suggested_action(q.text, answer, suggested_action)
            case = ActionCaseResult(
                question=q.text,
                answer=answer,
                suggested_action=suggested_action,
                latency_ms=latency_ms,
                relevance=rel,
                actionability=act,
                appropriateness=app,
                action_passed=judge_passed,
            )

        results.cases.append(case)
        if case.passed:
            results.passed += 1
        if suggested_action:
            results.total_with_actions += 1

        if verbose:
            status = "PASS" if case.passed else "FAIL"
            print(f"  {status} {q.text[:60]}")

    results.compute_aggregates()
    return results


def print_summary(results: ActionEvalResults) -> None:
    """Print action evaluation summary."""
    passed = results.pass_rate >= 0.80
    status = "PASS" if passed else "FAIL"

    print("\nAction Quality Evaluation (LLM Judge)")
    print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>=80.0% SLO) {status}")
    print(
        f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}, "
        f"Avg latency: {results.avg_latency_ms:.0f}ms"
    )
    print(f"Actions: {results.total_with_actions} cases with actions ({results.action_pass_rate * 100:.0f}% pass)")
    if results.total_with_actions > 0:
        print(
            f"Action Metrics: rel={results.avg_relevance:.2f} "
            f"act={results.avg_actionability:.2f} app={results.avg_appropriateness:.2f}"
        )

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        print(f"\nFailed Cases ({len(failed)})\n")

        for i, c in enumerate(failed[:10], 1):
            print(f"{i}. {c.question[:60]}")
            if c.errors:
                print(f"   Error: {'; '.join(c.errors)}")
            elif c.suggested_action:
                print(
                    f"   Action: rel={c.relevance:.2f} "
                    f"act={c.actionability:.2f} app={c.appropriateness:.2f}"
                )
                print(f"   Suggested: {c.suggested_action[:100]}...")
            print()

        if len(failed) > 10:
            print(f"... and {len(failed) - 10} more failures")


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run action quality evaluation using LLM Judge."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_action_eval(limit=limit, verbose=verbose))


if __name__ == "__main__":
    typer.run(main)
