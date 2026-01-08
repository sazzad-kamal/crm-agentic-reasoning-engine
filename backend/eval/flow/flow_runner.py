"""Flow evaluation runner - tests conversation paths."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.table import Table

from backend.agent.followup.tree import get_expected_answer, get_paths_for_role
from backend.eval.base import console, print_eval_header
from backend.eval.models import FlowEvalResults, FlowResult, FlowStepResult
from backend.eval.parallel import calculate_p95_latency
from backend.eval.ragas_judge import evaluate_single

logger = logging.getLogger(__name__)

# Lock for thread-safe console output
_console_lock = threading.Lock()


def _detect_expected_company(question: str) -> str | None:
    """
    Detect company mentioned in question text.

    Scans question for known company names from CRM database.
    Returns company_id if found, None otherwise.
    """
    from backend.agent.datastore import get_datastore

    ds = get_datastore()

    # Get all company names from CRM cache
    ds._build_company_cache()
    if not ds._company_names_cache:
        return None

    # Scan question for company names (case-insensitive)
    question_lower = question.lower()
    for name, company_id in ds._company_names_cache.items():
        if name in question_lower:
            return company_id

    return None


def judge_answer(
    question: str,
    answer: str,
    contexts: list[str],
    reference_answer: str | None = None,
    verbose: bool = False,
    timeout: int = 120,
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

        return {
            "relevance": result["answer_relevancy"],
            "faithfulness": result["faithfulness"],
            "context_precision": result["context_precision"],
            "context_recall": result.get("context_recall", 0.0),
            "answer_correctness": result.get("answer_correctness", 0.0),
            "explanation": "",
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

        # Get routing info from agent result
        actual_company_id = result.get("resolved_company_id")
        actual_intent = result.get("intent")

        # Detect expected company from question text
        expected_company = _detect_expected_company(question)

        # Company correct if:
        # - No company expected (None) -> True (don't penalize)
        # - Company expected and matches actual -> True
        # - Company expected but doesn't match -> False
        company_correct = (expected_company is None) or (actual_company_id == expected_company)

        # Get context chunks for RAGAS evaluation
        account_chunks = result.get("account_chunks", [])

        # Account RAG is conditional on intent and company_id
        # Check if it was actually invoked by looking at the result
        account_rag_invoked = bool(result.get("account_context_answer"))

        # Get expected answer for answer_correctness metric
        expected_answer = get_expected_answer(question)

        # Initialize RAGAS metrics
        relevance = 0.0
        faithfulness = 0.0
        answer_correctness = 0.0
        account_precision = 0.0
        account_recall = 0.0
        explanation = ""

        if use_judge and has_answer:
            # Evaluate account RAG if it was invoked
            if account_rag_invoked:
                account_result = judge_answer(
                    question, answer, account_chunks, reference_answer=expected_answer, verbose=verbose
                )
                account_precision = account_result.get("context_precision", 0.0)
                account_recall = account_result.get("context_recall", 0.0)
                relevance = account_result.get("relevance", 0.0)
                faithfulness = account_result.get("faithfulness", 0.0)
                answer_correctness = account_result.get("answer_correctness", 0.0)
            else:
                # If no account RAG, evaluate answer quality without context
                general_result = judge_answer(
                    question, answer, [], reference_answer=expected_answer, verbose=verbose
                )
                relevance = general_result.get("relevance", 0.0)
                faithfulness = general_result.get("faithfulness", 0.0)
                answer_correctness = general_result.get("answer_correctness", 0.0)

        return FlowStepResult(
            question=question,
            answer=answer,
            latency_ms=latency_ms,
            has_answer=has_answer,
            has_sources=len(sources) > 0,
            expected_company_id=expected_company,
            actual_company_id=actual_company_id,
            company_correct=company_correct,
            actual_intent=actual_intent,
            intent_correct=actual_intent is not None,
            relevance_score=relevance,
            faithfulness_score=faithfulness,
            answer_correctness_score=answer_correctness,
            account_precision_score=account_precision,
            account_recall_score=account_recall,
            account_rag_invoked=account_rag_invoked,
            judge_explanation=explanation,
            error=None,
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


def test_flow(path: list[str], path_id: int, use_judge: bool = True, verbose: bool = False) -> FlowResult:
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
    config_table.add_row("Total paths in tree", str(len(all_paths)))
    config_table.add_row("Paths to test", str(len(paths_to_test)))
    config_table.add_row("Questions per path", str(len(paths_to_test[0]) if paths_to_test else 0))
    config_table.add_row("Using LLM Judge", "Yes" if use_judge else "No")
    config_table.add_row("Concurrency", f"{concurrency} flows in parallel")
    console.print(config_table)
    console.print()

    completed = 0
    total = len(paths_to_test)
    results: list[FlowResult] = []

    console.print(f"[cyan]Starting {total} flows with {concurrency} workers...[/cyan]")

    # Use ThreadPoolExecutor for parallel flow execution
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

                with _console_lock:
                    completed += 1
                    status_color = "green" if result.success else "red"
                    status = "PASS" if result.success else "FAIL"
                    console.print(
                        f"[dim][{completed}/{total}][/dim] Path {path_id + 1}: "
                        f"[{status_color}]{status}[/{status_color}] ({result.total_latency_ms}ms)"
                    )
            except Exception as e:
                with _console_lock:
                    completed += 1
                    console.print(f"[red]Path {path_id + 1} raised exception: {e}[/red]")
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

    # Calculate routing accuracy
    # Only count questions that have expected company (skip questions without company context)
    steps_with_expected_company = [s for s in all_steps if s.expected_company_id is not None]
    company_sample_count = len(steps_with_expected_company)
    if company_sample_count > 0:
        company_correct_count = sum(1 for s in steps_with_expected_company if s.company_correct)
        company_extraction_accuracy = company_correct_count / company_sample_count
    else:
        company_extraction_accuracy = 0.0  # Will show as N/A

    intent_correct_count = sum(1 for s in all_steps if s.intent_correct)
    intent_accuracy = intent_correct_count / len(all_steps) if all_steps else 0.0

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
        company_extraction_accuracy=company_extraction_accuracy,
        company_sample_count=company_sample_count,
        intent_accuracy=intent_accuracy,
        avg_relevance=avg_relevance,
        avg_faithfulness=avg_faithfulness,
        avg_answer_correctness=avg_answer_correctness,
        avg_account_precision=avg_account_precision,
        avg_account_recall=avg_account_recall,
        account_sample_count=len(steps_with_account),
        total_latency_ms=total_latency,
        avg_latency_per_question_ms=avg_latency,
        p95_latency_ms=p95_latency,
        wall_clock_ms=wall_clock_ms,
        failed_paths=failed_paths,
        all_results=results,
    )
