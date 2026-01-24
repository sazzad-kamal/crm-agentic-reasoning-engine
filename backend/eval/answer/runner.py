"""Answer node evaluation runner."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml

if TYPE_CHECKING:
    import duckdb

from backend.agent.answer.answerer import call_answer_chain, extract_suggested_action
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.eval.answer.judge import judge_suggested_action
from backend.eval.answer.models import CaseResult, EvalMode, EvalResults, Question
from backend.eval.shared.ragas import evaluate_single

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent.parent / "shared" / "questions.yaml"


def load_questions() -> list[Question]:
    """Load questions from shared YAML file."""
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)
    return [Question(**item) for item in data.get("questions", [])]


def _eval_case(
    question: Question, conn: duckdb.DuckDBPyConnection, eval_mode: EvalMode = EvalMode.BOTH
) -> CaseResult:
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
                eval_mode=eval_mode,
                errors=errors,
            )

        # Step 2: Generate answer
        raw_answer = call_answer_chain(question.text, sql_results={"rows": sql_results})
        answer, suggested_action = extract_suggested_action(raw_answer)

        # Step 3: RAGAS evaluation (skip if action-only mode)
        ragas: dict = {"faithfulness": 0.0, "answer_relevancy": 0.0, "answer_correctness": 0.0}
        if eval_mode in (EvalMode.ANSWER, EvalMode.BOTH):
            contexts = [json.dumps(sql_results, default=str)] if sql_results else []
            ref_answer = question.expected_answer if question.expected_answer else None
            ragas = evaluate_single(question.text, answer, contexts, reference_answer=ref_answer)

        # Step 4: Action judge (skip if answer-only mode)
        action_metrics = (False, 0.0, 0.0, 0.0, "")
        if eval_mode in (EvalMode.ACTION, EvalMode.BOTH) and suggested_action:
            action_metrics = judge_suggested_action(question.text, answer, suggested_action)

        latency = int((time.time() - start) * 1000)
        return CaseResult(
            question=question.text,
            answer=answer,
            suggested_action=suggested_action,
            latency_ms=latency,
            eval_mode=eval_mode,
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
            eval_mode=eval_mode,
            errors=[f"Error: {e}"],
        )


def run_answer_eval(
    limit: int | None = None, verbose: bool = False, eval_mode: EvalMode = EvalMode.BOTH
) -> EvalResults:
    """Run answer node evaluation."""
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    results = EvalResults(total=len(questions))
    conn = get_connection()

    for q in questions:
        case = _eval_case(q, conn, eval_mode=eval_mode)
        results.cases.append(case)
        if case.passed:
            results.passed += 1
        if verbose:
            status = "PASS" if case.passed else "FAIL"
            print(f"  {status} {q.text[:60]}")

    results.compute_aggregates()
    return results


def print_summary(results: EvalResults, eval_mode: EvalMode = EvalMode.BOTH) -> None:
    """Print evaluation summary."""
    passed = results.pass_rate >= 0.80
    status = "PASS" if passed else "FAIL"

    mode_label = f" ({eval_mode.value} only)" if eval_mode != EvalMode.BOTH else ""
    print(f"\nAnswer Node Evaluation{mode_label}")
    print(f"Pass Rate: {results.pass_rate * 100:.1f}% (>=80.0% SLO) {status}")
    print(
        f"Total: {results.total}, Passed: {results.passed}, Failed: {results.failed}, "
        f"Avg latency: {results.avg_latency_ms:.0f}ms"
    )
    if eval_mode in (EvalMode.ANSWER, EvalMode.BOTH):
        print(
            f"RAGAS: F={results.avg_faithfulness:.2f} R={results.avg_relevance:.2f} "
            f"C={results.avg_answer_correctness:.2f} ({results.ragas_pass_rate * 100:.0f}%)"
        )
    if eval_mode in (EvalMode.ACTION, EvalMode.BOTH):
        print(
            f"Action: rel={results.avg_action_relevance:.2f} act={results.avg_action_actionability:.2f} "
            f"app={results.avg_action_appropriateness:.2f} ({results.action_pass_rate * 100:.0f}%)"
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
                ragas_failed = c.faithfulness_score < 0.6 or c.relevance_score < 0.6
                action_failed = c.suggested_action and not c.action_passed

                if eval_mode in (EvalMode.ANSWER, EvalMode.BOTH) and ragas_failed:
                    print(f"   RAGAS: F={c.faithfulness_score:.2f} R={c.relevance_score:.2f}")
                    if c.answer:
                        print(f"   Answer: {c.answer[:100]}...")
                if eval_mode in (EvalMode.ACTION, EvalMode.BOTH) and action_failed:
                    print(
                        f"   Action: rel={c.action_relevance:.2f} "
                        f"act={c.action_actionability:.2f} app={c.action_appropriateness:.2f}"
                    )
                    if c.suggested_action:
                        print(f"   Suggested: {c.suggested_action[:100]}...")
            print()

        if len(failed) > 10:
            print(f"... and {len(failed) - 10} more failures")


def main(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of questions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    eval: str = typer.Option("both", "--eval", "-e", help="Eval mode: answer, action, or both"),
) -> None:
    """Run answer node evaluation."""
    logging.basicConfig(level=logging.WARNING)
    eval_mode = EvalMode(eval)
    results = run_answer_eval(limit=limit, verbose=verbose, eval_mode=eval_mode)
    print_summary(results, eval_mode=eval_mode)


if __name__ == "__main__":
    typer.run(main)
