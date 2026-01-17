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
from backend.eval.integration.tree import (
    get_all_paths,
    get_expected_answer,
    get_expected_rag,
)
from backend.eval.shared.callback import get_eval_capture, reset_eval_capture
from backend.eval.shared.formatting import console, print_eval_header
from backend.eval.shared.ragas import evaluate_single

logger = logging.getLogger(__name__)


def _create_failed_judge_result(explanation: str) -> dict:
    """Create a failed RAGAS judge result with zeroed metrics."""
    return {
        "relevance": 0.0,
        "faithfulness": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
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
            "context_precision": result["context_precision"],
            "context_recall": result.get("context_recall", 0.0),
            "answer_correctness": result.get("answer_correctness", 0.0),
            "explanation": f"RAGAS error: {ragas_error}" if ragas_error else "",
            "ragas_failed": ragas_error is not None,
            "nan_metrics": nan_metrics,  # Pass through for per-metric failure tracking
        }
    except TimeoutError:
        logger.warning(f"RAGAS judge timed out after {timeout}s")
        return _create_failed_judge_result(f"RAGAS timeout after {timeout}s")
    except Exception as e:
        logger.warning(f"RAGAS judge failed: {e}")
        return _create_failed_judge_result(f"RAGAS error: {e}")


def _invoke_agent(question: str, session_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke the agent graph and return state + eval capture.

    Returns:
        Tuple of (state, eval_data) where eval_data contains metrics captured out-of-band.
    """
    from backend.agent.graph import agent_graph, build_thread_config

    # Reset eval capture before invoke
    reset_eval_capture()

    state: dict[str, Any] = {"question": question, "session_id": session_id}
    config = build_thread_config(session_id)
    result = agent_graph.invoke(state, config=config)

    # Get captured eval data
    eval_data = get_eval_capture()

    return result, eval_data


class RagasMetrics(TypedDict):
    """RAGAS evaluation metrics."""

    relevance: float
    faithfulness: float
    answer_correctness: float
    account_precision: float
    account_recall: float
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
    all_contexts: list[str],
    account_chunks: list[str],
    account_rag_invoked: bool,
    expected_answer: str | None,
    eval_mode: str,
    verbose: bool,
) -> RagasMetrics:
    """Run RAGAS evaluation and return metrics dict."""
    metrics: RagasMetrics = {
        "relevance": 0.0,
        "faithfulness": 0.0,
        "answer_correctness": 0.0,
        "account_precision": 0.0,
        "account_recall": 0.0,
        "explanation": "",
        "ragas_metrics_total": 0,
        "ragas_metrics_failed": 0,
    }

    # 1. RAG precision/recall - only when account RAG was invoked
    if eval_mode in ("rag", "both") and account_rag_invoked and account_chunks:
        rag_result = judge_answer(
            question, answer, account_chunks, reference_answer=expected_answer, verbose=verbose
        )
        metrics["account_precision"] = rag_result.get("context_precision", 0.0)
        metrics["account_recall"] = rag_result.get("context_recall", 0.0)
        metrics["ragas_metrics_total"] = 2
        metrics["ragas_metrics_failed"] = _count_failed_metrics(
            rag_result, ("context_precision", "context_recall")
        )

    # 2. Answer quality (faithfulness, relevance, correctness) - all contexts
    if eval_mode in ("pipeline", "both") and all_contexts:
        pipeline_result = judge_answer(
            question, answer, all_contexts, reference_answer=expected_answer, verbose=verbose
        )
        metrics["relevance"] = pipeline_result.get("relevance", 0.0)
        metrics["faithfulness"] = pipeline_result.get("faithfulness", 0.0)
        metrics["answer_correctness"] = pipeline_result.get("answer_correctness", 0.0)
        metrics["explanation"] = pipeline_result.get("explanation", "")

        pipeline_failed = _count_failed_metrics(
            pipeline_result, ("answer_relevancy", "faithfulness", "answer_correctness")
        )
        metrics["ragas_metrics_total"] += 3
        metrics["ragas_metrics_failed"] += pipeline_failed

    return metrics


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
    eval_mode: str = "both",
) -> FlowStepResult:
    """
    Test a single question with conversation history.

    Args:
        question: The question to ask
        history: List of {question, answer} dicts for memory
        session_id: Session ID for the conversation
        use_judge: Whether to run LLM-as-judge evaluation

    Returns:
        FlowStepResult with answer and metrics
    """
    start_time = time.time()

    try:
        # Invoke agent synchronously
        result, eval_data = _invoke_agent(question=question, session_id=session_id)

        latency_ms = int((time.time() - start_time) * 1000)

        answer = result.get("answer", "")
        has_answer = bool(answer and len(answer) > 10)

        # Get eval-specific data from out-of-band capture
        sql_queries_total = eval_data.get("sql_queries_total", 0)
        sql_queries_success = eval_data.get("sql_queries_success", 0)
        account_chunks = eval_data.get("account_chunks", [])
        account_rag_invoked = eval_data.get("account_rag_invoked", False)
        sql_plan = eval_data.get("sql_plan")

        # Build all_contexts from sql_results (JSON stringified)
        all_contexts = []
        sql_results = result.get("sql_results", {})
        if sql_results:
            all_contexts.append(json.dumps(sql_results, indent=2, default=str))
        if account_context := result.get("rag_context", ""):
            all_contexts.append(account_context)

        # Check RAG detection accuracy (needs_rag decision from sql_plan vs expected)
        rag_decision_correct: bool | None = None
        expected_rag = get_expected_rag(question)
        if expected_rag is not None:
            # Get needs_rag from sql_plan (from eval capture), fallback to account_rag_invoked
            actual_needs_rag = sql_plan.needs_rag if sql_plan else account_rag_invoked
            rag_decision_correct = actual_needs_rag == expected_rag

        # Get expected answer for answer_correctness metric
        expected_answer = get_expected_answer(question)

        # Run RAGAS evaluation if enabled
        ragas: RagasMetrics = {
            "relevance": 0.0, "faithfulness": 0.0, "answer_correctness": 0.0,
            "account_precision": 0.0, "account_recall": 0.0, "explanation": "",
            "ragas_metrics_total": 0, "ragas_metrics_failed": 0,
        }
        if use_judge and has_answer:
            ragas = _evaluate_ragas(
                question, answer, all_contexts, account_chunks,
                account_rag_invoked, expected_answer, eval_mode, verbose
            )

        return FlowStepResult(
            question=question,
            answer=answer,
            latency_ms=latency_ms,
            has_answer=has_answer,
            sql_queries_total=sql_queries_total,
            sql_queries_success=sql_queries_success,
            relevance_score=ragas["relevance"],
            faithfulness_score=ragas["faithfulness"],
            answer_correctness_score=ragas["answer_correctness"],
            account_precision_score=ragas["account_precision"],
            account_recall_score=ragas["account_recall"],
            account_rag_invoked=account_rag_invoked,
            rag_decision_correct=rag_decision_correct,
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
    eval_mode: str = "both",
) -> FlowResult:
    """
    Test a complete conversation flow (sequence of questions with memory).

    Args:
        path: List of questions in order
        path_id: ID for this path
        use_judge: Whether to run LLM-as-judge evaluation
        verbose: Show detailed RAGAS output
        eval_mode: RAGAS mode - 'rag', 'pipeline', or 'both'

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
        step_result = test_single_question(question, history, session_id, use_judge, verbose, eval_mode)
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

    # Account RAG metrics
    steps_with_account = [s for s in all_steps if s.account_rag_invoked]
    avg_account_precision = sum(s.account_precision_score for s in steps_with_account) / len(steps_with_account) if steps_with_account else 0.0
    avg_account_recall = sum(s.account_recall_score for s in steps_with_account) / len(steps_with_account) if steps_with_account else 0.0

    # SQL metrics
    sql_total = sum(s.sql_queries_total for s in all_steps)
    sql_success = sum(s.sql_queries_success for s in all_steps)
    sql_success_rate = sql_success / sql_total if sql_total > 0 else 1.0

    steps_with_assertions = [s for s in all_steps if s.sql_data_validated is not None]
    sql_data_passed = sum(1 for s in steps_with_assertions if s.sql_data_validated)
    sql_data_count = len(steps_with_assertions)
    sql_data_success_rate = sql_data_passed / sql_data_count if sql_data_count > 0 else 1.0

    # RAG detection accuracy
    steps_with_rag_expected = [s for s in all_steps if s.rag_decision_correct is not None]
    rag_detection_correct = sum(1 for s in steps_with_rag_expected if s.rag_decision_correct)
    rag_detection_count = len(steps_with_rag_expected)
    rag_detection_rate = rag_detection_correct / rag_detection_count if rag_detection_count > 0 else 1.0

    total_latency = sum(r.total_latency_ms for r in results)

    return {
        "total_paths": len(all_paths),
        "paths_tested": len(results),
        "paths_passed": paths_passed,
        "paths_failed": len(results) - paths_passed,
        "total_questions": total_questions,
        "questions_passed": questions_passed,
        "questions_failed": total_questions - questions_passed,
        "sql_success_rate": sql_success_rate,
        "sql_query_count": sql_total,
        "sql_data_success_rate": sql_data_success_rate,
        "sql_data_count": sql_data_count,
        "avg_relevance": avg_relevance,
        "avg_faithfulness": avg_faithfulness,
        "avg_answer_correctness": avg_answer_correctness,
        "avg_account_precision": avg_account_precision,
        "avg_account_recall": avg_account_recall,
        "account_sample_count": len(steps_with_account),
        "rag_invoked_count": sum(1 for s in all_steps if s.account_rag_invoked),
        "rag_detection_rate": rag_detection_rate,
        "rag_detection_count": rag_detection_count,
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
    eval_mode: str = "both",
) -> FlowEvalResults:
    """
    Run the flow evaluation on all paths using ThreadPoolExecutor.

    Args:
        max_paths: Limit number of paths to test (None = all)
        verbose: Print detailed output
        use_judge: Whether to run LLM-as-judge evaluation
        concurrency: Number of flows to run in parallel (default 5)
        eval_mode: RAGAS mode - 'rag', 'pipeline', or 'both'

    Returns:
        FlowEvalResults with aggregated metrics
    """
    eval_start_time = time.time()

    # Generate all paths
    all_paths = get_all_paths()
    paths_to_test = all_paths[:max_paths] if max_paths else all_paths

    # Print header
    print_eval_header(
        "[bold blue]Conversation Flow Evaluation[/bold blue]",
        "Testing multi-turn conversation paths with RAGAS metrics",
    )

    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value")
    config_table.add_row("Paths to test", str(len(paths_to_test)))
    config_table.add_row("RAGAS Mode", eval_mode)
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
                executor.submit(test_flow, path, i, use_judge, verbose, eval_mode): i
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
