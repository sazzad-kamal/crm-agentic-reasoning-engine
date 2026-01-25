"""Text quality evaluation runner using RAGAS metrics."""

from __future__ import annotations

import json
import logging
from typing import cast

import typer

from backend.agent.fetch.sql.connection import get_connection
from backend.eval.answer.shared import generate_answer, load_questions
from backend.eval.answer.text.models import SLO_TEXT_PASS_RATE, TextCaseResult, TextEvalResults
from backend.eval.answer.text.ragas import evaluate_single

logger = logging.getLogger(__name__)


def run_text_eval(limit: int | None = None, verbose: bool = False) -> TextEvalResults:
    """Run text quality evaluation using RAGAS metrics."""
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = TextEvalResults(total=len(questions))
    conn = get_connection()

    for q in questions:
        answer, _, sql_results, error = generate_answer(q, conn)

        if error:
            case = TextCaseResult(
                question=q.text,
                answer="",
                errors=[error],
            )
        else:
            # Run RAGAS evaluation
            contexts = [json.dumps(sql_results, default=str)] if sql_results else []
            ref_answer = q.expected_answer
            ragas = evaluate_single(q.text, answer, contexts, reference_answer=ref_answer)
            nan_metrics = cast(list[str], ragas.get("nan_metrics", []))

            case = TextCaseResult(
                question=q.text,
                answer=answer,
                faithfulness_score=cast(float, ragas["faithfulness"]),
                relevance_score=cast(float, ragas["answer_relevancy"]),
                answer_correctness_score=cast(float, ragas.get("answer_correctness", 0.0)),
                ragas_metrics_total=3 if ref_answer else 2,
                ragas_metrics_failed=len(nan_metrics),
            )

        results.cases.append(case)
        if case.passed:
            results.passed += 1

        if verbose:
            status = "PASS" if case.passed else "FAIL"
            print(f"  {status} {q.text[:60]}")

    results.compute_aggregates()
    return results


def print_summary(results: TextEvalResults) -> None:
    """Print text evaluation summary."""
    passed = results.pass_rate >= SLO_TEXT_PASS_RATE
    status = "PASS" if passed else "FAIL"

    print("\nText Quality Evaluation (RAGAS)")
    print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>={SLO_TEXT_PASS_RATE * 100:.1f}% SLO) {status}")
    print(f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}")
    print(
        f"RAGAS: F={results.avg_faithfulness:.2f} R={results.avg_relevance:.2f} "
        f"C={results.avg_answer_correctness:.2f}"
    )
    ragas_success = results.ragas_metrics_total - results.ragas_metrics_failed
    print(
        f"RAGAS Reliability: {ragas_success}/{results.ragas_metrics_total} "
        f"({results.ragas_success_rate * 100:.1f}%)"
    )

    # Failed cases
    failed = [c for c in results.cases if not c.passed]
    if failed:
        print(f"\nFailed Cases ({len(failed)})\n")

        for i, c in enumerate(failed[:10], 1):
            print(f"{i}. {c.question[:60]}")
            if c.errors:
                print(f"   Error: {'; '.join(c.errors)}")
            else:
                print(f"   RAGAS: F={c.faithfulness_score:.2f} R={c.relevance_score:.2f} C={c.answer_correctness_score:.2f}")
                if c.answer:
                    ans = c.answer[:100] + "..." if len(c.answer) > 100 else c.answer
                    print(f"   Answer: {ans}")
            print()

        if len(failed) > 10:
            print(f"... and {len(failed) - 10} more failures")


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run text quality evaluation using RAGAS metrics."""
    logging.basicConfig(level=logging.WARNING)
    print_summary(run_text_eval(limit=limit, verbose=verbose))


if __name__ == "__main__":
    typer.run(main)
