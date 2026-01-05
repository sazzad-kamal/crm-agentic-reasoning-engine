"""Flow evaluation runner - tests conversation paths."""

from __future__ import annotations

import asyncio
import logging
import time

from rich.table import Table

from backend.agent.question_tree import get_paths_for_role
from backend.agent.eval.base import console, print_eval_header
from backend.agent.eval.parallel import calculate_p95_latency
from backend.agent.eval.shared import run_llm_judge
from backend.agent.eval.models import FlowStepResult, FlowResult, FlowEvalResults
from backend.agent.eval.prompts import FLOW_JUDGE_SYSTEM, FLOW_JUDGE_PROMPT

logger = logging.getLogger(__name__)


def judge_answer(
    question: str,
    answer: str,
    context: str = "",
) -> dict:
    """Judge an answer using LLM."""
    prompt = FLOW_JUDGE_PROMPT.format(
        question=question,
        answer=answer[:1000],  # Truncate long answers
        context=context[:500] if context else "First question in conversation",
    )

    result = run_llm_judge(prompt, FLOW_JUDGE_SYSTEM)

    if "error" in result:
        logger.warning(f"Judge failed: {result['error']}")
        return {"relevance": 0, "grounded": 0, "explanation": f"Judge error: {result['error']}"}

    return {
        "relevance": result.get("answer_relevance", 0),
        "grounded": result.get("answer_grounded", 0),
        "explanation": result.get("explanation", ""),
    }


async def test_single_question(
    question: str,
    history: list[dict],
    session_id: str,
    use_judge: bool = True,
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
    from backend.agent.graph import run_agent

    start_time = time.time()

    try:
        # Run the agent in a thread pool for true parallelism
        result = await asyncio.to_thread(
            run_agent,
            question=question,
            mode="auto",
            session_id=session_id,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        has_answer = bool(answer and len(answer) > 10)

        # Build context from history for judge
        context = ""
        if history:
            context = "\n".join(
                [
                    f"Q: {h['question']}\nA: {h['answer'][:200]}..."
                    for h in history[-2:]  # Last 2 turns
                ]
            )

        # Run LLM judge if enabled and we have an answer
        relevance = 0
        grounded = 0
        explanation = ""

        if use_judge and has_answer:
            judge_result = await asyncio.to_thread(judge_answer, question, answer, context)
            relevance = judge_result.get("relevance", 0)
            grounded = judge_result.get("grounded", 0)
            explanation = judge_result.get("explanation", "")

        return FlowStepResult(
            question=question,
            answer=answer,
            latency_ms=latency_ms,
            has_answer=has_answer,
            has_sources=len(sources) > 0,
            relevance_score=relevance,
            grounded_score=grounded,
            judge_explanation=explanation,
            error=None,
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
            relevance_score=0,
            grounded_score=0,
            judge_explanation=f"Error: {e}",
            error=str(e),
        )


async def test_flow(path: list[str], path_id: int, use_judge: bool = True) -> FlowResult:
    """
    Test a complete conversation flow (sequence of questions with memory).

    Args:
        path: List of questions in order
        path_id: ID for this path
        use_judge: Whether to run LLM-as-judge evaluation

    Returns:
        FlowResult with all step results
    """
    session_id = f"flow_eval_{path_id}_{int(time.time())}"
    history: list[dict] = []
    steps: list[FlowStepResult] = []
    total_latency = 0
    success = True

    for question in path:
        step_result = await test_single_question(question, history, session_id, use_judge)
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


async def run_flow_eval(
    max_paths: int | None = None,
    verbose: bool = False,
    use_judge: bool = True,
    concurrency: int = 5,
) -> FlowEvalResults:
    """
    Run the flow evaluation on all paths.

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
        "Testing multi-turn conversation paths with LLM-as-judge",
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

    # Semaphore to limit concurrent flows
    semaphore = asyncio.Semaphore(concurrency)
    results: list[FlowResult] = []
    completed = 0
    total = len(paths_to_test)
    lock = asyncio.Lock()

    async def run_with_semaphore(path: list[str], path_id: int) -> FlowResult:
        nonlocal completed
        async with semaphore:
            result = await test_flow(path, path_id, use_judge)
            async with lock:
                completed += 1
                status_color = "green" if result.success else "red"
                status = "PASS" if result.success else "FAIL"
                console.print(
                    f"[dim][{completed}/{total}][/dim] Path {path_id + 1}: "
                    f"[{status_color}]{status}[/{status_color}] ({result.total_latency_ms}ms)"
                )
                if verbose or not result.success:
                    for j, step in enumerate(result.steps):
                        step_color = "green" if step.passed else "red"
                        status_icon = "PASS" if step.passed else "FAIL"
                        console.print(f"  Q{j + 1}: {step.question[:50]}...")
                        console.print(
                            f"      [{step_color}][{status_icon}][/{step_color}] "
                            f"R={step.relevance_score} G={step.grounded_score} | {step.latency_ms}ms"
                        )
                        if step.judge_explanation and verbose:
                            console.print(
                                f"      [dim]Judge: {step.judge_explanation[:80]}...[/dim]"
                            )
                        if step.error:
                            console.print(f"      [red]ERROR: {step.error}[/red]")
            return result

    # Run all flows in parallel
    tasks = [run_with_semaphore(path, i) for i, path in enumerate(paths_to_test)]
    console.print(f"[cyan]Starting {len(tasks)} flows...[/cyan]")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for exceptions in results
    actual_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            console.print(f"[red]Path {i + 1} raised exception: {r}[/red]")
            actual_results.append(
                FlowResult(
                    path_id=i,
                    questions=paths_to_test[i],
                    steps=[],
                    total_latency_ms=0,
                    success=False,
                    error=str(r),
                )
            )
        else:
            actual_results.append(r)
    results = actual_results

    # Aggregate results
    paths_passed = sum(1 for r in results if r.success)
    paths_failed = len(results) - paths_passed

    total_questions = sum(len(r.steps) for r in results)
    questions_passed = sum(sum(1 for s in r.steps if s.passed) for r in results)
    questions_failed = total_questions - questions_passed

    # Calculate judge score averages
    all_steps = [s for r in results for s in r.steps]
    avg_relevance = sum(s.relevance_score for s in all_steps) / len(all_steps) if all_steps else 0
    avg_grounded = sum(s.grounded_score for s in all_steps) / len(all_steps) if all_steps else 0

    total_latency = sum(r.total_latency_ms for r in results)
    avg_latency = total_latency / total_questions if total_questions > 0 else 0

    # Calculate P95 latency per question
    step_latencies = [s.latency_ms for s in all_steps]
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
        avg_relevance=avg_relevance,
        avg_grounded=avg_grounded,
        total_latency_ms=total_latency,
        avg_latency_per_question_ms=avg_latency,
        p95_latency_ms=p95_latency,
        wall_clock_ms=wall_clock_ms,
        failed_paths=failed_paths,
        all_results=results,
    )
