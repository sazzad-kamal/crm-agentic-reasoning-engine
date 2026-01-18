"""Evaluation runner - tests conversation paths."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypedDict

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from backend.eval.integration.models import FlowEvalResults, FlowResult, FlowStepResult
from backend.eval.integration.tree import get_all_paths, get_expected_answer
from backend.eval.shared.formatting import console
from backend.eval.shared.ragas import evaluate_single

logger = logging.getLogger(__name__)


def _create_failed_judge_result(explanation: str) -> dict:
    """Create a failed RAGAS judge result with zeroed metrics."""
    return {
        "relevance": 0.0,
        "faithfulness": 0.0,
        "answer_correctness": 0.0,
        "explanation": explanation,
        "ragas_failed": True,
        "nan_metrics": [],
    }


def judge_answer(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
    verbose: bool = False,
    timeout: int = 180,
) -> dict:
    """Judge an answer using RAGAS metrics with timeout."""
    try:
        # Run RAGAS evaluation with timeout to prevent hanging
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                evaluate_single, question, answer, contexts,
                reference_answer=reference_answer, verbose=verbose
            )
            result = future.result(timeout=timeout)

        # Check if RAGAS itself reported an error
        ragas_error = result.get("error")
        nan_metrics: list[str] = result.get("nan_metrics", [])  # type: ignore[assignment]
        return {
            "relevance": result["answer_relevancy"],
            "faithfulness": result["faithfulness"],
            "answer_correctness": result.get("answer_correctness", 0.0),
            "explanation": f"RAGAS error: {ragas_error}" if ragas_error else "",
            "ragas_failed": ragas_error is not None,
            "nan_metrics": nan_metrics,
        }
    except TimeoutError:
        logger.warning(f"RAGAS judge timed out after {timeout}s")
        return _create_failed_judge_result(f"RAGAS timeout after {timeout}s")
    except Exception as e:
        logger.warning(f"RAGAS judge failed: {e}")
        return _create_failed_judge_result(f"RAGAS error: {e}")


def _invoke_agent(question: str, session_id: str | None = None) -> dict[str, Any]:
    """Invoke the agent graph and return result."""
    from backend.agent.graph import agent_graph, build_thread_config

    state: dict[str, Any] = {"question": question, "session_id": session_id}
    config = build_thread_config(session_id)
    return agent_graph.invoke(state, config=config)


class RagasMetrics(TypedDict):
    """RAGAS evaluation metrics."""

    relevance: float
    faithfulness: float
    answer_correctness: float
    explanation: str
    ragas_metrics_total: int
    ragas_metrics_failed: int


def _count_failed_metrics(result: dict, metric_names: tuple[str, ...]) -> int:
    """Count failed metrics from a RAGAS result."""
    if result.get("ragas_failed"):
        return len(metric_names)
    nan_metrics = result.get("nan_metrics", [])
    return sum(1 for m in nan_metrics if m in metric_names)


def _evaluate_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str | None,
    verbose: bool,
) -> RagasMetrics:
    """Run RAGAS evaluation for answer quality metrics."""
    if not contexts:
        return {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "explanation": "No context available",
            "ragas_metrics_total": 0,
            "ragas_metrics_failed": 0,
        }

    result = judge_answer(question, answer, contexts, reference_answer=expected_answer, verbose=verbose)
    return {
        "relevance": result.get("relevance", 0.0),
        "faithfulness": result.get("faithfulness", 0.0),
        "answer_correctness": result.get("answer_correctness", 0.0),
        "explanation": result.get("explanation", ""),
        "ragas_metrics_total": 3,
        "ragas_metrics_failed": _count_failed_metrics(
            result, ("answer_relevancy", "faithfulness", "answer_correctness")
        ),
    }


def _create_error_step_result(question: str, latency_ms: int, error: str) -> FlowStepResult:
    """Create a FlowStepResult for error cases."""
    return FlowStepResult(
        question=question,
        answer="",
        latency_ms=latency_ms,
        has_answer=False,
        relevance_score=0.0,
        faithfulness_score=0.0,
        answer_correctness_score=0.0,
        judge_explanation=error,
        error=error,
    )


def test_single_question(
    question: str,
    history: list[dict],
    session_id: str,
    use_judge: bool = True,
    verbose: bool = False,
) -> FlowStepResult:
    """
    Test a single question with conversation history.

    Args:
        question: The question to ask
        history: List of {question, answer} dicts for memory
        session_id: Session ID for the conversation
        use_judge: Whether to run LLM-as-judge evaluation
        verbose: Show detailed RAGAS output

    Returns:
        FlowStepResult with answer and metrics
    """
    start_time = time.time()

    try:
        result = _invoke_agent(question=question, session_id=session_id)
        latency_ms = int((time.time() - start_time) * 1000)

        answer = result.get("answer", "")
        has_answer = bool(answer and len(answer) > 10)

        # Build contexts from result
        contexts = []
        sql_results = result.get("sql_results", {})
        if sql_results:
            contexts.append(json.dumps(sql_results, indent=2, default=str))
        if rag_context := result.get("rag_context", ""):
            contexts.append(rag_context)

        # Get expected answer for answer_correctness metric
        expected_answer = get_expected_answer(question)

        # Run RAGAS evaluation if enabled
        ragas: RagasMetrics = {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "explanation": "",
            "ragas_metrics_total": 0,
            "ragas_metrics_failed": 0,
        }
        if use_judge and has_answer:
            ragas = _evaluate_ragas(question, answer, contexts, expected_answer, verbose)

        return FlowStepResult(
            question=question,
            answer=answer,
            latency_ms=latency_ms,
            has_answer=has_answer,
            relevance_score=ragas["relevance"],
            faithfulness_score=ragas["faithfulness"],
            answer_correctness_score=ragas["answer_correctness"],
            judge_explanation=ragas["explanation"],
            error=None,
            ragas_metrics_total=ragas["ragas_metrics_total"],
            ragas_metrics_failed=ragas["ragas_metrics_failed"],
        )

    except TimeoutError:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Timeout testing question '{question}' after 120s")
        return _create_error_step_result(question, latency_ms, "Timeout after 120s")
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error testing question '{question}': {e}")
        return _create_error_step_result(question, latency_ms, f"Error: {e}")


def test_flow(
    path: list[str],
    path_id: int,
    use_judge: bool = True,
    verbose: bool = False,
) -> FlowResult:
    """
    Test a complete conversation flow (sequence of questions with memory).

    Args:
        path: List of questions in order
        path_id: ID for this path
        use_judge: Whether to run LLM-as-judge evaluation
        verbose: Show detailed RAGAS output

    Returns:
        FlowResult with all step results
    """
    session_id = f"flow_eval_{path_id}_{int(time.time())}"
    history: list[dict] = []
    steps: list[FlowStepResult] = []
    total_latency = 0
    success = True

    # Questions within a flow must be sequential (memory dependency)
    for question in path:
        step_result = test_single_question(question, history, session_id, use_judge, verbose)
        steps.append(step_result)
        total_latency += step_result.latency_ms

        if not step_result.passed:
            success = False

        history.append(
            {
                "question": question,
                "answer": step_result.answer,
            }
        )

    return FlowResult(
        path_id=path_id,
        questions=path,
        steps=steps,
        total_latency_ms=total_latency,
        success=success,
        error=steps[-1].error if steps and steps[-1].error else None,
    )


def _aggregate_metrics(
    results: list[FlowResult], all_paths: list[list[str]], eval_start_time: float
) -> dict[str, Any]:
    """Aggregate metrics from flow results into a summary dict."""
    paths_passed = sum(1 for r in results if r.success)
    total_questions = sum(len(r.steps) for r in results)
    questions_passed = sum(sum(1 for s in r.steps if s.passed) for r in results)

    all_steps = [s for r in results for s in r.steps]

    # RAGAS averages
    avg_relevance = sum(s.relevance_score for s in all_steps) / len(all_steps) if all_steps else 0.0
    avg_faithfulness = sum(s.faithfulness_score for s in all_steps) / len(all_steps) if all_steps else 0.0
    avg_answer_correctness = sum(s.answer_correctness_score for s in all_steps) / len(all_steps) if all_steps else 0.0

    total_latency = sum(r.total_latency_ms for r in results)

    return {
        "total_paths": len(all_paths),
        "paths_tested": len(results),
        "paths_passed": paths_passed,
        "paths_failed": len(results) - paths_passed,
        "total_questions": total_questions,
        "questions_passed": questions_passed,
        "questions_failed": total_questions - questions_passed,
        "avg_relevance": avg_relevance,
        "avg_faithfulness": avg_faithfulness,
        "avg_answer_correctness": avg_answer_correctness,
        "ragas_metrics_total": sum(s.ragas_metrics_total for s in all_steps),
        "ragas_metrics_failed": sum(s.ragas_metrics_failed for s in all_steps),
        "total_latency_ms": total_latency,
        "avg_latency_per_question_ms": total_latency / total_questions if total_questions > 0 else 0,
        "wall_clock_ms": int((time.time() - eval_start_time) * 1000),
        "failed_paths": [r for r in results if not r.success],
        "all_results": results,
    }


def run_flow_eval(
    max_paths: int | None = None,
    verbose: bool = False,
    use_judge: bool = True,
    concurrency: int = 5,
) -> FlowEvalResults:
    """
    Run the flow evaluation on all paths using ThreadPoolExecutor.

    Args:
        max_paths: Limit number of paths to test (None = all)
        verbose: Print detailed output
        use_judge: Whether to run LLM-as-judge evaluation
        concurrency: Number of flows to run in parallel (default 5)

    Returns:
        FlowEvalResults with aggregated metrics
    """
    eval_start_time = time.time()

    # Generate all paths
    all_paths = get_all_paths()
    paths_to_test = all_paths[:max_paths] if max_paths else all_paths

    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value")
    config_table.add_row("Paths to test", str(len(paths_to_test)))
    config_table.add_row("Concurrency", str(concurrency))
    console.print(config_table)
    console.print()

    total = len(paths_to_test)
    results: list[FlowResult] = []

    # Use ThreadPoolExecutor for parallel flow execution with clean progress bar
    with Progress(
        TextColumn("[cyan]Evaluating paths"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("", total=total)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all flows
            futures = {
                executor.submit(test_flow, path, i, use_judge, verbose): i
                for i, path in enumerate(paths_to_test)
            }

            # Process results as they complete
            for future in as_completed(futures):
                path_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(
                        FlowResult(
                            path_id=path_id,
                            questions=paths_to_test[path_id],
                            steps=[],
                            total_latency_ms=0,
                            success=False,
                            error=str(e),
                        )
                    )
                progress.advance(task)

    # Sort results by path_id for consistent ordering
    results.sort(key=lambda r: r.path_id)

    # Aggregate and return results
    metrics = _aggregate_metrics(results, all_paths, eval_start_time)
    return FlowEvalResults(**metrics)
