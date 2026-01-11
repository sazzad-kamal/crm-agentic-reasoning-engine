"""Evaluation runner - tests conversation paths."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from backend.agent.followup.tree import (
    get_expected_answer,
    get_expected_rag,
    get_expected_sql_results,
    get_paths_for_role,
    validate_sql_results,
)
from backend.eval.formatting import console, print_eval_header
from backend.eval.judge import evaluate_single
from backend.eval.models import FlowEvalResults, FlowResult, FlowStepResult
from backend.eval.parallel import calculate_p95_latency

logger = logging.getLogger(__name__)


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
        nan_metrics = result.get("nan_metrics", [])
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
        return {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
            "explanation": f"RAGAS timeout after {timeout}s",
            "ragas_failed": True,
            "nan_metrics": [],  # Not applicable for timeout (ragas_failed handles it)
        }
    except Exception as e:
        logger.warning(f"RAGAS judge failed: {e}")
        return {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
            "explanation": f"RAGAS error: {e}",
            "ragas_failed": True,
            "nan_metrics": [],  # Not applicable for exception (ragas_failed handles it)
        }


def _invoke_agent(question: str, session_id: str | None = None) -> dict[str, Any]:
    """Invoke the agent graph and return state."""
    from backend.agent.graph import agent_graph, build_thread_config

    state: dict[str, Any] = {"question": question, "session_id": session_id, "sources": []}
    config = build_thread_config(session_id)
    return agent_graph.invoke(state, config=config)  # type: ignore[no-any-return]


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
        result = _invoke_agent(question=question, session_id=session_id)

        latency_ms = int((time.time() - start_time) * 1000)

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        has_answer = bool(answer and len(answer) > 10)

        # Get SQL execution stats
        sql_queries_total = result.get("sql_queries_total", 0)
        sql_queries_success = result.get("sql_queries_success", 0)

        # Get RAG chunks for precision/recall evaluation
        account_chunks = result.get("account_chunks", [])

        # Account RAG is called automatically when entity IDs are resolved
        # Use the explicit flag set by fetch_account_node (tracks actual invocations)
        account_rag_invoked = result.get("account_rag_invoked", False)

        # Build all_contexts from sql_results (JSON stringified)
        all_contexts = []
        sql_results = result.get("sql_results", {})
        if sql_results:
            all_contexts.append(json.dumps(sql_results, indent=2, default=str))
        if account_context := result.get("account_context_answer", ""):
            all_contexts.append(account_context)

        # Validate SQL results against expected (for all questions with assertions)
        sql_data_validated: bool | None = None
        sql_data_errors: list[str] | None = None
        if sql_results:
            passed, errors = validate_sql_results(question, sql_results)
            # Set validated if we have assertions for this question (SQL-only OR RAG)
            if get_expected_sql_results(question) is not None:
                sql_data_validated = passed
                sql_data_errors = errors if errors else None

        # Check RAG detection accuracy (needs_rag decision from slot planner vs expected)
        rag_decision_correct: bool | None = None
        expected_rag = get_expected_rag(question)
        if expected_rag is not None:
            # Compare actual needs_rag decision (use account_rag_invoked as proxy if needs_rag not in result)
            actual_needs_rag = result.get("needs_rag", account_rag_invoked)
            rag_decision_correct = actual_needs_rag == expected_rag

        # Get expected answer for answer_correctness metric
        expected_answer = get_expected_answer(question)

        # Initialize RAGAS metrics
        relevance = 0.0
        faithfulness = 0.0
        answer_correctness = 0.0
        account_precision = 0.0
        account_recall = 0.0
        explanation = ""
        ragas_metrics_total = 0
        ragas_metrics_failed = 0

        if use_judge and has_answer:
            # 1. RAG precision/recall - only when account RAG was invoked
            # Skip if eval_mode is "pipeline" (only want answer quality metrics)
            if eval_mode in ("rag", "both") and account_rag_invoked and account_chunks:
                rag_result = judge_answer(
                    question, answer, account_chunks, reference_answer=expected_answer, verbose=verbose
                )
                account_precision = rag_result.get("context_precision", 0.0)
                account_recall = rag_result.get("context_recall", 0.0)
                # Track precision + recall (2 metrics)
                ragas_metrics_total = 2
                # Count failures: timeout/error sets ragas_failed=True, else check nan_metrics
                if rag_result.get("ragas_failed"):
                    ragas_metrics_failed = 2  # All RAG metrics failed
                else:
                    nan_metrics = rag_result.get("nan_metrics", [])
                    ragas_metrics_failed = sum(1 for m in nan_metrics if m in ("context_precision", "context_recall"))

            # 2. Answer quality (faithfulness, relevance, correctness) - all contexts
            # Skip if eval_mode is "rag" (only want precision/recall metrics)
            if eval_mode in ("pipeline", "both") and all_contexts:
                pipeline_result = judge_answer(
                    question, answer, all_contexts, reference_answer=expected_answer, verbose=verbose
                )
                relevance = pipeline_result.get("relevance", 0.0)
                faithfulness = pipeline_result.get("faithfulness", 0.0)
                answer_correctness = pipeline_result.get("answer_correctness", 0.0)
                explanation = pipeline_result.get("explanation", "")
                # Track only relevance + faithfulness + correctness (3 metrics) in pipeline mode
                if eval_mode == "pipeline":
                    ragas_metrics_total = 3
                    # Count failures: timeout/error sets ragas_failed=True, else check nan_metrics
                    if pipeline_result.get("ragas_failed"):
                        ragas_metrics_failed = 3  # All pipeline metrics failed
                    else:
                        nan_metrics = pipeline_result.get("nan_metrics", [])
                        ragas_metrics_failed = sum(1 for m in nan_metrics if m in ("answer_relevancy", "faithfulness", "answer_correctness"))
                else:
                    # both mode: count all 5 metrics (2 from rag + 3 from pipeline)
                    ragas_metrics_total += 3
                    # Count failures: timeout/error sets ragas_failed=True, else check nan_metrics
                    if pipeline_result.get("ragas_failed"):
                        ragas_metrics_failed += 3  # All pipeline metrics failed
                    else:
                        nan_metrics = pipeline_result.get("nan_metrics", [])
                        ragas_metrics_failed += sum(1 for m in nan_metrics if m in ("answer_relevancy", "faithfulness", "answer_correctness"))

        return FlowStepResult(
            question=question,
            answer=answer,
            latency_ms=latency_ms,
            has_answer=has_answer,
            has_sources=len(sources) > 0,
            sql_queries_total=sql_queries_total,
            sql_queries_success=sql_queries_success,
            sql_data_validated=sql_data_validated,
            sql_data_errors=sql_data_errors,
            relevance_score=relevance,
            faithfulness_score=faithfulness,
            answer_correctness_score=answer_correctness,
            account_precision_score=account_precision,
            account_recall_score=account_recall,
            account_rag_invoked=account_rag_invoked,
            rag_decision_correct=rag_decision_correct,
            judge_explanation=explanation,
            error=None,
            ragas_metrics_total=ragas_metrics_total,
            ragas_metrics_failed=ragas_metrics_failed,
        )

    except TimeoutError:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Timeout testing question '{question}' after 120s")
        return FlowStepResult(
            question=question,
            answer="",
            latency_ms=latency_ms,
            has_answer=False,
            has_sources=False,
            relevance_score=0.0,
            faithfulness_score=0.0,
            answer_correctness_score=0.0,
            judge_explanation="Timeout after 120s",
            error="Timeout after 120s",
        )
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error testing question '{question}': {e}")
        return FlowStepResult(
            question=question,
            answer="",
            latency_ms=latency_ms,
            has_answer=False,
            has_sources=False,
            relevance_score=0.0,
            faithfulness_score=0.0,
            answer_correctness_score=0.0,
            judge_explanation=f"Error: {e}",
            error=str(e),
        )


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
    all_paths = get_paths_for_role()
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

    # Aggregate results
    paths_passed = sum(1 for r in results if r.success)
    paths_failed = len(results) - paths_passed

    total_questions = sum(len(r.steps) for r in results)
    questions_passed = sum(sum(1 for s in r.steps if s.passed) for r in results)
    questions_failed = total_questions - questions_passed

    # Calculate RAGAS metric averages
    all_steps = [s for r in results for s in r.steps]
    avg_relevance = sum(s.relevance_score for s in all_steps) / len(all_steps) if all_steps else 0.0
    avg_faithfulness = sum(s.faithfulness_score for s in all_steps) / len(all_steps) if all_steps else 0.0
    avg_answer_correctness = sum(s.answer_correctness_score for s in all_steps) / len(all_steps) if all_steps else 0.0

    # Calculate account RAG metrics (only count steps where it was invoked)
    steps_with_account = [s for s in all_steps if s.account_rag_invoked]

    avg_account_precision = sum(s.account_precision_score for s in steps_with_account) / len(steps_with_account) if steps_with_account else 0.0
    avg_account_recall = sum(s.account_recall_score for s in steps_with_account) / len(steps_with_account) if steps_with_account else 0.0

    # Calculate SQL success rate (query execution)
    sql_total = sum(s.sql_queries_total for s in all_steps)
    sql_success = sum(s.sql_queries_success for s in all_steps)
    sql_success_rate = sql_success / sql_total if sql_total > 0 else 1.0  # 1.0 if no queries

    # Calculate SQL data validation success rate
    steps_with_assertions = [s for s in all_steps if s.sql_data_validated is not None]
    sql_data_passed = sum(1 for s in steps_with_assertions if s.sql_data_validated)
    sql_data_count = len(steps_with_assertions)
    sql_data_success_rate = sql_data_passed / sql_data_count if sql_data_count > 0 else 1.0

    # Calculate RAG detection accuracy (needs_rag decision matches expected)
    steps_with_rag_expected = [s for s in all_steps if s.rag_decision_correct is not None]
    rag_detection_correct = sum(1 for s in steps_with_rag_expected if s.rag_decision_correct)
    rag_detection_count = len(steps_with_rag_expected)
    rag_detection_rate = rag_detection_correct / rag_detection_count if rag_detection_count > 0 else 1.0

    # Calculate RAGAS reliability (per-metric, not per-call)
    # Sum up all metrics evaluated and all metrics that failed across all steps
    ragas_metrics_total = sum(s.ragas_metrics_total for s in all_steps)
    ragas_metrics_failed = sum(s.ragas_metrics_failed for s in all_steps)

    total_latency = sum(r.total_latency_ms for r in results)
    avg_latency = total_latency / total_questions if total_questions > 0 else 0

    # Calculate P95 latency per question
    step_latencies: list[float | int] = [s.latency_ms for s in all_steps]
    p95_latency = calculate_p95_latency(step_latencies)

    failed_paths = [r for r in results if not r.success]
    wall_clock_ms = int((time.time() - eval_start_time) * 1000)

    return FlowEvalResults(
        total_paths=len(all_paths),
        paths_tested=len(results),
        paths_passed=paths_passed,
        paths_failed=paths_failed,
        total_questions=total_questions,
        questions_passed=questions_passed,
        questions_failed=questions_failed,
        sql_success_rate=sql_success_rate,
        sql_query_count=sql_total,
        sql_data_success_rate=sql_data_success_rate,
        sql_data_count=sql_data_count,
        avg_relevance=avg_relevance,
        avg_faithfulness=avg_faithfulness,
        avg_answer_correctness=avg_answer_correctness,
        avg_account_precision=avg_account_precision,
        avg_account_recall=avg_account_recall,
        account_sample_count=len(steps_with_account),
        rag_invoked_count=sum(1 for s in all_steps if s.account_rag_invoked),
        rag_detection_rate=rag_detection_rate,
        rag_detection_count=rag_detection_count,
        ragas_metrics_total=ragas_metrics_total,
        ragas_metrics_failed=ragas_metrics_failed,
        total_latency_ms=total_latency,
        avg_latency_per_question_ms=avg_latency,
        p95_latency_ms=p95_latency,
        wall_clock_ms=wall_clock_ms,
        failed_paths=failed_paths,
        all_results=results,
    )
