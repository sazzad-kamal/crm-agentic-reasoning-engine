"""Action quality evaluation runner using LLM Judge."""

from __future__ import annotations

import logging
from typing import Any

import typer
from dotenv import load_dotenv

load_dotenv()

from backend.agent.fetch.sql.connection import get_connection
from backend.eval.answer.action.judge import judge_suggested_action
from backend.eval.answer.action.models import (
    SLO_ACTION_PASS_RATE,
    ActionCaseResult,
    ActionEvalResults,
)
from backend.eval.answer.shared import generate_answer, load_questions

logger = logging.getLogger(__name__)


def run_action_eval(limit: int | None = None) -> ActionEvalResults:
    """Run action quality evaluation using LLM Judge."""
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = ActionEvalResults(total=len(questions))
    conn = get_connection()

    for idx, q in enumerate(questions, 1):
        answer, suggested_action, _, error = generate_answer(q, conn)

        kwargs: dict[str, Any] = {
            "answer": answer,
            "suggested_action": suggested_action,
        }

        if error:
            kwargs.update(answer="", suggested_action=None, errors=[error])
        elif q.expected_action and suggested_action:
            # Action expected and produced: judge it
            try:
                judge_passed, rel, act, app, _ = judge_suggested_action(
                    q.text, answer, suggested_action
                )
                kwargs.update(
                    relevance=rel,
                    actionability=act,
                    appropriateness=app,
                    action_passed=judge_passed,
                )
            except Exception as e:
                logger.warning(f"Judge evaluation failed: {e}")
                kwargs["errors"] = [f"Judge failed: {e}"]
        elif not q.expected_action and not suggested_action:
            # Correct silence: not expected and not produced
            kwargs["action_passed"] = True
        # else: action_missing or spurious_action — action_passed defaults to False

        case = ActionCaseResult(
            question=q.text, expected_action=q.expected_action, **kwargs
        )

        results.cases.append(case)
        status = "PASS" if case.passed else "FAIL"
        print(f"  [{idx}/{results.total}] {status} {q.text[:50]}")

    results.compute_aggregates()
    return results


def print_summary(results: ActionEvalResults) -> None:
    """Print action evaluation summary."""
    passed = results.pass_rate >= SLO_ACTION_PASS_RATE
    status = "PASS" if passed else "FAIL"

    print("\nAction Quality Evaluation (LLM Judge)")
    print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>={SLO_ACTION_PASS_RATE * 100:.1f}% SLO) {status}")
    print(f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}")
    print(
        f"  Action expected + correct:   {results.action_expected_passed} passed, "
        f"{results.action_expected_failed} failed (judged)"
    )
    if results.action_expected_passed + results.action_expected_failed > 0:
        print(
            f"  Action Metrics: rel={results.avg_relevance:.2f} "
            f"act={results.avg_actionability:.2f} app={results.avg_appropriateness:.2f}"
        )
    print()
    print(f"  Action expected + missing:   {results.action_missing} failed")
    print(f"  Spurious action:             {results.spurious_action} failed")
    print(f"  No action expected (quiet):  {results.correct_silence} passed")
    if results.error_count > 0:
        print(f"  Errors:                      {results.error_count} failed")

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
            print(f"{i}. {c.question[:60]}")
            if c.expected_action and not c.suggested_action:
                print("   Reason: Action expected but not produced")
            elif not c.expected_action and c.suggested_action:
                print("   Reason: Spurious action produced")
                print(f"   Suggested: {c.suggested_action[:100]}...")
            elif c.suggested_action:
                print(
                    f"   Action: rel={c.relevance:.2f} "
                    f"act={c.actionability:.2f} app={c.appropriateness:.2f}"
                )
                print(f"   Suggested: {c.suggested_action[:100]}...")
            if c.answer:
                ans = c.answer[:100] + "..." if len(c.answer) > 100 else c.answer
                print(f"   Answer: {ans}")
            print()


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
) -> None:
    """Run action quality evaluation using LLM Judge."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_action_eval(limit=limit))


if __name__ == "__main__":
    typer.run(main)
