"""Evaluation runner - tests conversation paths."""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import time
import traceback
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

# Fix Windows asyncio cleanup issues with httpx/RAGAS
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined,unused-ignore]

import typer

from backend.eval.answer.text.ragas import evaluate_single
from backend.eval.integration.langsmith import get_latency_percentages
from backend.eval.integration.models import (
    SLO_FLOW_PATH_PASS_RATE,
    FlowEvalResults,
    FlowResult,
    FlowStepResult,
)
from backend.eval.integration.tree import get_all_paths, get_expected_answer, get_tree_stats

logger = logging.getLogger(__name__)

MIN_ANSWER_LENGTH = 10


def _invoke_agent(question: str, session_id: str | None = None) -> dict[str, Any]:
    """Invoke the agent graph and return result."""
    from backend.agent.graph import agent_graph, build_thread_config

    state: dict[str, Any] = {"question": question, "session_id": session_id}
    config = build_thread_config(session_id)
    return cast(dict[str, Any], agent_graph.invoke(state, config=config))


def test_single_question(question: str, session_id: str, use_judge: bool = True) -> FlowStepResult:
    """Test a single question and return answer with metrics."""
    start_time = time.time()

    try:
        result = _invoke_agent(question=question, session_id=session_id)
        latency_ms = int((time.time() - start_time) * 1000)

        answer = result.get("answer", "")
        has_answer = bool(answer and len(answer) > MIN_ANSWER_LENGTH)

        # Build contexts and run RAGAS if enabled
        relevance = correctness = 0.0
        ragas_total = ragas_failed = 0

        sql_results = result.get("sql_results", {})
        if use_judge and has_answer and sql_results:
            contexts = [json.dumps(sql_results, default=str)]
            expected = get_expected_answer(question)
            try:
                ragas = evaluate_single(question, answer, contexts, expected or "")
                nan_metrics = cast(list[str], ragas.get("nan_metrics", []))
                relevance = cast(float, ragas["answer_relevancy"])
                correctness = cast(float, ragas["answer_correctness"])
                ragas_total = 2
                ragas_failed = len(nan_metrics)
            except Exception as e:
                logger.warning(f"RAGAS failed: {e}")

        return FlowStepResult(
            question=question,
            answer=answer,
            latency_ms=latency_ms,
            has_answer=has_answer,
            relevance_score=relevance,
            answer_correctness_score=correctness,
            error=None,
            ragas_metrics_total=ragas_total,
            ragas_metrics_failed=ragas_failed,
        )

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error testing question '{question}': {e}")
        return FlowStepResult(
            question=question, answer="", latency_ms=latency_ms, has_answer=False,
            relevance_score=0.0, answer_correctness_score=0.0, error=str(e),
        )


def test_flow(path: list[str], path_id: int, use_judge: bool = True) -> FlowResult:
    """Test a conversation flow (sequential questions with memory)."""
    session_id = f"flow_eval_{path_id}_{int(time.time())}"
    steps: list[FlowStepResult] = []
    total_latency = 0
    success = True
    first_error: str | None = None

    for question in path:
        step_result = test_single_question(question, session_id, use_judge)
        steps.append(step_result)
        total_latency += step_result.latency_ms

        if not step_result.passed:
            success = False
            if first_error is None and step_result.error:
                first_error = step_result.error

    return FlowResult(
        path_id=path_id, questions=path, steps=steps,
        total_latency_ms=total_latency, success=success, error=first_error,
    )


def run_flow_eval(max_paths: int | None = None, use_judge: bool = True) -> FlowEvalResults:
    """Run flow evaluation on all paths."""
    all_paths = get_all_paths()
    paths_to_test = all_paths[:max_paths] if max_paths else all_paths
    total = len(paths_to_test)

    results: list[FlowResult] = []
    for i, path in enumerate(paths_to_test):
        result = test_flow(path, i, use_judge)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{i+1}/{total}] {status} Path {i+1}: {path[0][:40]}...")

    # Aggregate metrics
    all_steps = [s for r in results for s in r.steps]
    paths_passed = sum(1 for r in results if r.success)
    total_questions = sum(len(r.steps) for r in results)
    questions_passed = sum(sum(1 for s in r.steps if s.passed) for r in results)
    total_latency = sum(r.total_latency_ms for r in results)

    return FlowEvalResults(
        total_paths=len(all_paths),
        paths_tested=len(results),
        paths_passed=paths_passed,
        paths_failed=len(results) - paths_passed,
        total_questions=total_questions,
        questions_passed=questions_passed,
        questions_failed=total_questions - questions_passed,
        avg_relevance=sum(s.relevance_score for s in all_steps) / len(all_steps) if all_steps else 0.0,
        avg_answer_correctness=sum(s.answer_correctness_score for s in all_steps) / len(all_steps) if all_steps else 0.0,
        ragas_metrics_total=sum(s.ragas_metrics_total for s in all_steps),
        ragas_metrics_failed=sum(s.ragas_metrics_failed for s in all_steps),
        total_latency_ms=total_latency,
        avg_latency_per_question_ms=total_latency / total_questions if total_questions > 0 else 0,
        all_results=results,
    )


def print_summary(results: FlowEvalResults, latency_pcts: dict[str, float] | None = None) -> bool:
    """Print evaluation summary. Returns True if passed."""
    passed = results.path_pass_rate >= SLO_FLOW_PATH_PASS_RATE
    status = "PASS" if passed else "FAIL"

    print("\nFlow Evaluation Summary")
    print(f"Pass Rate: {results.path_pass_rate:.1%} (>={SLO_FLOW_PATH_PASS_RATE:.1%} SLO) {status}")
    print(f"Paths: {results.paths_passed}/{results.paths_tested}, Questions: {results.questions_passed}/{results.total_questions}")
    print(f"Avg Relevance: {results.avg_relevance:.2f}, Correctness: {results.avg_answer_correctness:.2f}")

    ragas_ok = results.ragas_metrics_total - results.ragas_metrics_failed
    print(f"RAGAS: {ragas_ok}/{results.ragas_metrics_total} ({results.ragas_success_rate:.1%})")

    if latency_pcts:
        print(f"LangSmith: fetch={latency_pcts.get('fetch', 0):.1%}, answer={latency_pcts.get('answer', 0):.1%}")

    # Failed paths
    failed = [r for r in results.all_results if not r.success]
    if failed:
        print(f"\nFailed Paths ({len(failed)})")
        for r in failed[:5]:
            print(f"  Path {r.path_id + 1}: {r.questions[0][:50]}...")
            if r.error:
                print(f"    Error: {r.error}")

    return passed


def _run_eval(limit: int | None) -> None:
    """Run the flow evaluation."""
    eval_start_time = time.time()

    stats = get_tree_stats()
    print("\nQuestion Tree Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    try:
        results = run_flow_eval(max_paths=limit, use_judge=True)
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        traceback.print_exc()
        return

    elapsed_minutes = int((time.time() - eval_start_time) / 60) + 1
    latency_pcts = get_latency_percentages(minutes_ago=max(elapsed_minutes, 5))
    print_summary(results, latency_pcts=latency_pcts)


def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    _run_eval(limit=limit)


if __name__ == "__main__":
    typer.run(main)
