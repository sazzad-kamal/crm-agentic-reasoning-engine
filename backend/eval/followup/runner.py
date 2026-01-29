"""Followup suggestion evaluation runner using LLM Judge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import typer
from dotenv import load_dotenv

if TYPE_CHECKING:
    import duckdb

load_dotenv()

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.agent.followup.suggester import generate_follow_up_suggestions
from backend.eval.answer.shared.loader import generate_answer, load_questions
from backend.eval.followup.judge import judge_followup_suggestions
from backend.eval.followup.models import (
    SLO_FOLLOWUP_ANSWERABILITY,
    SLO_FOLLOWUP_PASS_RATE,
    FollowupCaseResult,
    FollowupEvalResults,
)

logger = logging.getLogger(__name__)


def _check_answerability(
    suggestions: list[str], conn: duckdb.DuckDBPyConnection,
) -> tuple[int, float]:
    """Check how many suggestions are answerable via SQL.

    Returns (answerable_count, answerability_ratio).
    """
    if not suggestions:
        return 0, 0.0

    answerable = 0
    for suggestion in suggestions:
        try:
            plan = get_sql_plan(suggestion)
            rows, error = execute_sql(plan.sql, conn)
            if not error and rows:
                answerable += 1
        except Exception:
            logger.debug(f"Answerability check failed for: {suggestion[:50]}")

    return answerable, answerable / len(suggestions)


def run_followup_eval(
    limit: int | None = None,
) -> FollowupEvalResults:
    """Run followup suggestion evaluation using LLM Judge."""
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = FollowupEvalResults(total=len(questions))
    conn = get_connection()

    for idx, q in enumerate(questions, 1):
        suggestions: list[str] = []
        errors: list[str] = []

        # Generate answer first (like action eval)
        answer, _, answer_error = generate_answer(q, conn)
        if answer_error:
            errors.append(f"Answer error: {answer_error}")

        if not errors:
            try:
                suggestions = generate_follow_up_suggestions(
                    question=q.text,
                    answer=answer,
                    use_hardcoded_tree=False,
                )
            except Exception as e:
                errors.append(f"Generation error: {e}")

        judge_kwargs: dict = {}
        if suggestions and not errors:
            try:
                passed, qrel, agrnd, div, explanation = judge_followup_suggestions(
                    q.text, suggestions, answer=answer
                )
                judge_kwargs = {
                    "passed": passed,
                    "question_relevance": qrel,
                    "answer_grounding": agrnd,
                    "diversity": div,
                    "explanation": explanation,
                }
            except Exception as e:
                logger.warning(f"Judge evaluation failed: {e}")
                errors.append(f"Judge failed: {e}")

        # Answerability check
        answerability_kwargs: dict = {}
        if suggestions and not errors:
            try:
                answerable_count, answerability = _check_answerability(suggestions, conn)
                answerability_kwargs = {
                    "answerable_count": answerable_count,
                    "answerability": answerability,
                }
            except Exception as e:
                logger.warning(f"Answerability check failed: {e}")

        case = FollowupCaseResult(
            question=q.text,
            answer=answer,
            suggestions=suggestions,
            errors=errors,
            **judge_kwargs,
            **answerability_kwargs,
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

    ans_passed = results.avg_answerability >= SLO_FOLLOWUP_ANSWERABILITY
    ans_status = "PASS" if ans_passed else "FAIL"

    print("\nFollowup Suggestion Evaluation (LLM Judge)")
    print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>={SLO_FOLLOWUP_PASS_RATE * 100:.1f}% SLO) {status}")
    print(f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}")
    print(
        f"  Followup Metrics: qrel={results.avg_question_relevance:.2f}"
        f" agrnd={results.avg_answer_grounding:.2f}"
        f" div={results.avg_diversity:.2f}"
    )
    print(
        f"  Answerability: {results.avg_answerability * 100:.1f}%"
        f" (>={SLO_FOLLOWUP_ANSWERABILITY * 100:.1f}% SLO) {ans_status}"
    )

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
            print(f"   Scores: qrel={c.question_relevance:.2f} agrnd={c.answer_grounding:.2f} div={c.diversity:.2f}")
            print(f"   Answerability: {c.answerable_count}/{len(c.suggestions)}")
            if c.explanation:
                print(f"   Judge: {c.explanation}")
            if c.suggestions:
                for s in c.suggestions:
                    print(f"   - {s}")
            print()


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
) -> None:
    """Run followup suggestion evaluation using LLM Judge."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_followup_eval(limit=limit))


if __name__ == "__main__":
    typer.run(main)
