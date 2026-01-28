"""Evaluation runner - tests conversation paths."""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import time
import traceback
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()

# Fix Windows asyncio cleanup issues with httpx/RAGAS
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined,unused-ignore]

import typer

from backend.eval.answer.text.ragas import evaluate_single
from backend.eval.integration.langsmith import get_latency_percentages
from backend.eval.integration.models import (
    SLO_FLOW_PASS_RATE,
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
            ragas_metrics_total=ragas_total,
            ragas_metrics_failed=ragas_failed,
        )

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error testing question '{question}': {e}")
        return FlowStepResult(
            question=question, answer="", latency_ms=latency_ms, has_answer=False,
            errors=[str(e)],
        )


def test_flow(path: list[str], path_id: int, use_judge: bool = True) -> FlowResult:
    """Test a conversation flow (sequential questions with memory)."""
    session_id = f"flow_eval_{path_id}_{int(time.time())}"
    steps: list[FlowStepResult] = []
    total_latency = 0
    success = True

    for question in path:
        step_result = test_single_question(question, session_id, use_judge)
        steps.append(step_result)
        total_latency += step_result.latency_ms

        if not step_result.passed:
            success = False

    return FlowResult(
        path_id=path_id, questions=path, steps=steps,
        total_latency_ms=total_latency, success=success,
    )


def run_flow_eval(max_paths: int | None = None, use_judge: bool = True) -> FlowEvalResults:
    """Run flow evaluation on all paths."""
    all_paths = get_all_paths()
    paths_to_test = all_paths[:max_paths] if max_paths else all_paths

    results = FlowEvalResults(total=len(paths_to_test))
    total = len(paths_to_test)

    for idx, path in enumerate(paths_to_test, 1):
        result = test_flow(path, idx - 1, use_judge)
        results.cases.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{idx}/{total}] {status} Path {idx}: {path[0][:40]}...")

    results.compute_aggregates()
    return results


def print_summary(results: FlowEvalResults, latency_pcts: dict[str, float] | None = None) -> bool:
    """Print evaluation summary. Returns True if passed."""
    passed = results.pass_rate >= SLO_FLOW_PASS_RATE
    status = "PASS" if passed else "FAIL"

    total_questions = sum(len(c.steps) for c in results.cases)
    questions_passed = sum(sum(1 for s in c.steps if s.passed) for c in results.cases)

    print("\nFlow Evaluation Summary")
    print(f"Pass Rate: {results.pass_rate:.1%} (>={SLO_FLOW_PASS_RATE:.1%} SLO) {status}")
    print(f"Paths: {results.passed}/{results.total}, Questions: {questions_passed}/{total_questions}")
    print(f"Avg Relevance: {results.avg_relevance:.2f}, Correctness: {results.avg_answer_correctness:.2f}")

    ragas_ok = results.ragas_metrics_total - results.ragas_metrics_failed
    print(f"RAGAS: {ragas_ok}/{results.ragas_metrics_total} ({results.ragas_success_rate:.1%})")

    if latency_pcts:
        print(f"LangSmith: fetch={latency_pcts.get('fetch', 0):.1%}, answer={latency_pcts.get('answer', 0):.1%}")

    # Failed paths
    failed = [c for c in results.cases if not c.success]
    if failed:
        print(f"\nFailed Paths ({len(failed)})")
        for c in failed[:5]:
            print(f"  Path {c.path_id + 1}: {c.questions[0][:50]}...")
            step_errors = [e for s in c.steps for e in s.errors]
            if step_errors:
                print(f"    Error: {step_errors[0]}")

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
