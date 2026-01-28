"""Text quality evaluation runner using RAGAS metrics."""

from __future__ import annotations

import json
import logging
from typing import cast

import typer
from dotenv import load_dotenv

load_dotenv()

from backend.agent.fetch.sql.connection import get_connection
from backend.eval.answer.shared import generate_answer, load_questions
from backend.eval.answer.text.models import SLO_TEXT_PASS_RATE, TextCaseResult, TextEvalResults
from backend.eval.answer.text.ragas import RAGAS_METRICS_COUNT, evaluate_single

logger = logging.getLogger(__name__)


def run_text_eval(limit: int | None = None) -> TextEvalResults:
    """Run text quality evaluation using RAGAS metrics."""
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = TextEvalResults(total=len(questions))
    conn = get_connection()

    total = len(questions)
    for idx, q in enumerate(questions, 1):
        answer, sql_results, error = generate_answer(q, conn)

        if error:
            case = TextCaseResult(
                question=q.text,
                answer="",
                errors=[error],
            )
        elif not sql_results:
            case = TextCaseResult(
                question=q.text,
                answer=answer,
                errors=["No SQL results - skipping RAGAS"],
            )
        else:
            # Run RAGAS evaluation
            contexts = [json.dumps(sql_results, default=str)]
            try:
                ragas = evaluate_single(q.text, answer, contexts, q.expected_answer)
                nan_metrics = cast(list[str], ragas.get("nan_metrics", []))

                case = TextCaseResult(
                    question=q.text,
                    answer=answer,
                    answer_correctness_score=cast(float, ragas["answer_correctness"]),
                    answer_relevancy_score=cast(float, ragas["answer_relevancy"]),
                    ragas_metrics_total=RAGAS_METRICS_COUNT,
                    ragas_metrics_failed=len(nan_metrics),
                )
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed: {e}")
                case = TextCaseResult(
                    question=q.text,
                    answer=answer,
                    errors=[f"RAGAS failed: {e}"],
                )

        results.cases.append(case)
        status = "PASS" if case.passed else "FAIL"
        print(f"  [{idx}/{total}] {status} {q.text[:50]}")

    results.compute_aggregates()
    return results


def print_summary(results: TextEvalResults) -> None:
    """Print text evaluation summary."""
    passed = results.pass_rate >= SLO_TEXT_PASS_RATE
    status = "PASS" if passed else "FAIL"

    print("\nText Quality Evaluation (RAGAS)")
    print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>={SLO_TEXT_PASS_RATE * 100:.1f}% SLO) {status}")
    print(f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}")
    print(f"Avg Correctness: {results.avg_answer_correctness:.2f}, Relevancy: {results.avg_answer_relevancy:.2f}")
    ragas_success = results.ragas_metrics_total - results.ragas_metrics_failed
    print(
        f"RAGAS Reliability: {ragas_success}/{results.ragas_metrics_total} "
        f"({results.ragas_success_rate * 100:.1f}%)"
    )

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        print(f"\nFailed Cases ({len(failed)})\n")

        for i, c in enumerate(failed, 1):
            print(f"{i}. {c.question[:60]}")
            if c.errors:
                print(f"   Error: {'; '.join(c.errors)}")
            else:
                print(f"   Correctness: {c.answer_correctness_score:.2f}, Relevancy: {c.answer_relevancy_score:.2f}")
                if c.answer:
                    ans = c.answer[:100] + "..." if len(c.answer) > 100 else c.answer
                    print(f"   Answer: {ans}")
            print()


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
) -> None:
    """Run text quality evaluation using RAGAS metrics."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_text_eval(limit=limit))


if __name__ == "__main__":
    typer.run(main)
