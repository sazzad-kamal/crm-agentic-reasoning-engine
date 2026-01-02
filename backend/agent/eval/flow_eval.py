"""
Conversation flow evaluation - tests all paths through the question tree.

This eval simulates a user clicking through the demo, testing each possible
path with accumulating conversation memory.

Usage:
    python -m backend.agent.eval.flow_eval
    python -m backend.agent.eval.flow_eval --limit 10  # Quick test
    python -m backend.agent.eval.flow_eval --verbose  # Show all responses
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

# Load .env file before any other imports
from dotenv import load_dotenv

load_dotenv()

# Ensure collections exist before running eval
from backend.agent.eval.base import ensure_qdrant_collections

print("Checking Qdrant collections...")
ensure_qdrant_collections()
print()

import typer
from rich.table import Table
from rich.panel import Panel

from backend.agent.question_tree import generate_all_paths, get_tree_stats
from backend.agent.eval.base import (
    console,
    format_percentage,
    format_check_mark,
    print_eval_header,
    compare_to_baseline,
    save_baseline,
    print_baseline_comparison,
)
from backend.agent.eval.shared import (
    parse_json_response,
    print_slo_result,
    calculate_p95_latency,
    determine_exit_code,
    get_failed_slos,
    print_overall_result_panel,
)
from backend.agent.eval.prompts import FLOW_JUDGE_SYSTEM, FLOW_JUDGE_PROMPT
from backend.agent.eval.models import (
    SLO_FLOW_PATH_PASS_RATE,
    SLO_FLOW_QUESTION_PASS_RATE,
    SLO_FLOW_RELEVANCE,
    SLO_FLOW_GROUNDED,
    SLO_FLOW_AVG_LATENCY_MS,
    SLO_FLOW_P95_LATENCY_MS,
    FlowStepResult,
    FlowResult,
    FlowEvalResults,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Judge
# =============================================================================


def judge_answer(
    question: str,
    answer: str,
    context: str = "",
) -> dict:
    """Judge an answer using LLM."""
    from backend.agent.eval.llm_client import call_llm

    prompt = FLOW_JUDGE_PROMPT.format(
        question=question,
        answer=answer[:1000],  # Truncate long answers
        context=context[:500] if context else "First question in conversation",
    )

    try:
        response = call_llm(
            prompt=prompt,
            system_prompt=FLOW_JUDGE_SYSTEM,
            model="gpt-4o-mini",
            temperature=0,
        )

        result = parse_json_response(response)
        return {
            "relevance": result.get("answer_relevance", 0),
            "grounded": result.get("answer_grounded", 0),
            "explanation": result.get("explanation", ""),
        }
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return {"relevance": 0, "grounded": 0, "explanation": f"Judge error: {e}"}


# =============================================================================
# Flow Testing
# =============================================================================


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
        # (run_agent is synchronous, so we need asyncio.to_thread)
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
            # Run judge in thread pool for parallelism
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

        # Use the passed property which checks judge scores
        if not step_result.passed:
            success = False

        # Add to history for next question
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
    all_paths = generate_all_paths()
    paths_to_test = all_paths[:max_paths] if max_paths else all_paths

    # Print header using rich
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
                    f"[dim][{completed}/{total}][/dim] Path {path_id + 1}: [{status_color}]{status}[/{status_color}] ({result.total_latency_ms}ms)"
                )
                if verbose or not result.success:
                    for j, step in enumerate(result.steps):
                        step_color = "green" if step.passed else "red"
                        status_icon = "PASS" if step.passed else "FAIL"
                        console.print(f"  Q{j + 1}: {step.question[:50]}...")
                        console.print(
                            f"      [{step_color}][{status_icon}][/{step_color}] R={step.relevance_score} G={step.grounded_score} | {step.latency_ms}ms"
                        )
                        if step.judge_explanation and verbose:
                            console.print(
                                f"      [dim]Judge: {step.judge_explanation[:80]}...[/dim]"
                            )
                        if step.error:
                            console.print(f"      [red]ERROR: {step.error}[/red]")
            return result

    # Run all flows in parallel (limited by semaphore)
    tasks = [run_with_semaphore(path, i) for i, path in enumerate(paths_to_test)]
    console.print(f"[cyan]Starting {len(tasks)} flows...[/cyan]")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for exceptions in results
    actual_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            console.print(f"[red]Path {i + 1} raised exception: {r}[/red]")
            # Create a failed result
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

    eval_results = FlowEvalResults(
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

    return eval_results


def print_summary(results: FlowEvalResults):
    """Print a comprehensive summary of eval results with SLO status."""
    console.print()

    # ==========================================================================
    # Main Summary Table
    # ==========================================================================
    summary_table = Table(
        title="Flow Evaluation Summary", show_header=True, header_style="bold cyan"
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")
    summary_table.add_column("SLO", justify="right", style="dim")
    summary_table.add_column("Status", justify="center")

    # Path metrics
    path_pass_rate = results.path_pass_rate
    path_slo_pass = path_pass_rate >= SLO_FLOW_PATH_PASS_RATE
    summary_table.add_row(
        "Paths Tested",
        f"{results.paths_tested}/{results.total_paths}",
        "",
        "",
    )
    summary_table.add_row(
        "Path Pass Rate",
        format_percentage(path_pass_rate),
        f">={format_percentage(SLO_FLOW_PATH_PASS_RATE)}",
        format_check_mark(path_slo_pass),
    )
    summary_table.add_row("", "", "", "")  # Spacer

    # Question metrics
    q_pass_rate = results.question_pass_rate
    q_slo_pass = q_pass_rate >= SLO_FLOW_QUESTION_PASS_RATE
    summary_table.add_row(
        "Questions Total",
        str(results.total_questions),
        "",
        "",
    )
    summary_table.add_row(
        "Question Pass Rate",
        format_percentage(q_pass_rate),
        f">={format_percentage(SLO_FLOW_QUESTION_PASS_RATE)}",
        format_check_mark(q_slo_pass),
    )
    summary_table.add_row("", "", "", "")  # Spacer

    # Judge score metrics
    relevance_slo_pass = results.avg_relevance >= SLO_FLOW_RELEVANCE
    grounded_slo_pass = results.avg_grounded >= SLO_FLOW_GROUNDED
    summary_table.add_row(
        "[bold]LLM Judge Scores[/bold]",
        "",
        "",
        "",
    )
    summary_table.add_row(
        "  Relevance",
        format_percentage(results.avg_relevance),
        f">={format_percentage(SLO_FLOW_RELEVANCE)}",
        format_check_mark(relevance_slo_pass),
    )
    summary_table.add_row(
        "  Groundedness",
        format_percentage(results.avg_grounded),
        f">={format_percentage(SLO_FLOW_GROUNDED)}",
        format_check_mark(grounded_slo_pass),
    )
    summary_table.add_row("", "", "", "")  # Spacer

    # Latency metrics (tracked, not SLO)
    summary_table.add_row(
        "[bold]Latency[/bold]",
        "",
        "",
        "",
    )
    summary_table.add_row(
        "  Avg per Question",
        f"{results.avg_latency_per_question_ms:.0f}ms",
        "[dim]tracked[/dim]",
        "",
    )
    summary_table.add_row(
        "  P95 per Question",
        f"{results.p95_latency_ms:.0f}ms",
        "[dim]tracked[/dim]",
        "",
    )
    summary_table.add_row(
        "  Total",
        f"{results.total_latency_ms}ms",
        "",
        "",
    )

    # Wall clock time
    wall_secs = results.wall_clock_ms / 1000
    summary_table.add_row("", "", "", "")  # Spacer
    summary_table.add_row(
        "Wall Clock Time",
        f"{wall_secs:.1f}s ({wall_secs / 60:.1f} min)",
        "",
        "",
    )

    console.print(summary_table)

    # ==========================================================================
    # SLO Summary Panel
    # ==========================================================================
    slo_checks = [
        (
            "Path Pass Rate",
            path_slo_pass,
            format_percentage(path_pass_rate),
            f">={format_percentage(SLO_FLOW_PATH_PASS_RATE)}",
        ),
        (
            "Question Pass Rate",
            q_slo_pass,
            format_percentage(q_pass_rate),
            f">={format_percentage(SLO_FLOW_QUESTION_PASS_RATE)}",
        ),
        (
            "Relevance",
            relevance_slo_pass,
            format_percentage(results.avg_relevance),
            f">={format_percentage(SLO_FLOW_RELEVANCE)}",
        ),
        (
            "Groundedness",
            grounded_slo_pass,
            format_percentage(results.avg_grounded),
            f">={format_percentage(SLO_FLOW_GROUNDED)}",
        ),
    ]

    # Print SLO summary table
    all_slos_passed = print_slo_result(slo_checks)

    # ==========================================================================
    # Failed Paths Detail
    # ==========================================================================
    if results.failed_paths:
        console.print()
        failed_table = Table(
            title=f"Failed Paths ({len(results.failed_paths)} total, showing first 5)",
            show_header=True,
            header_style="bold yellow",
        )
        failed_table.add_column("Path", style="bold", width=6)
        failed_table.add_column("Question", width=50)
        failed_table.add_column("R", justify="center", width=3)
        failed_table.add_column("G", justify="center", width=3)
        failed_table.add_column("Latency", justify="right", width=8)
        failed_table.add_column("Issue", width=40)

        for fp in results.failed_paths[:5]:
            for i, step in enumerate(fp.steps):
                if not step.passed:
                    issue = (
                        step.judge_explanation[:40]
                        if step.judge_explanation
                        else (step.error or "Unknown")
                    )
                    failed_table.add_row(
                        str(fp.path_id) if i == 0 else "",
                        step.question[:48] + "..." if len(step.question) > 48 else step.question,
                        format_check_mark(step.relevance_score == 1),
                        format_check_mark(step.grounded_score == 1),
                        f"{step.latency_ms}ms",
                        issue,
                    )

        console.print(failed_table)

    # ==========================================================================
    # Overall Result
    # ==========================================================================
    failed_slo_names = get_failed_slos(slo_checks)
    all_passed = all_slos_passed and results.paths_failed == 0

    failure_reasons = []
    if results.paths_failed > 0:
        failure_reasons.append(f"{results.paths_failed} paths failed")
    if failed_slo_names:
        failure_reasons.append(f"{len(failed_slo_names)} SLOs not met: {', '.join(failed_slo_names)}")

    console.print()
    print_overall_result_panel(
        all_passed=all_passed,
        failure_reasons=failure_reasons,
        success_message=f"All {results.paths_tested} paths passed, all SLOs met",
    )


def save_results(results: FlowEvalResults, output_path: Path):
    """Save results to JSON file."""
    data = {
        "summary": {
            "total_paths": results.total_paths,
            "paths_tested": results.paths_tested,
            "paths_passed": results.paths_passed,
            "paths_failed": results.paths_failed,
            "path_pass_rate": results.path_pass_rate,
            "total_questions": results.total_questions,
            "questions_passed": results.questions_passed,
            "questions_failed": results.questions_failed,
            "question_pass_rate": results.question_pass_rate,
            "avg_relevance": results.avg_relevance,
            "avg_grounded": results.avg_grounded,
            "total_latency_ms": results.total_latency_ms,
            "avg_latency_per_question_ms": results.avg_latency_per_question_ms,
            "p95_latency_ms": results.p95_latency_ms,
            "wall_clock_ms": results.wall_clock_ms,
        },
        "slo_results": {
            "path_pass_rate": {
                "value": results.path_pass_rate,
                "target": SLO_FLOW_PATH_PASS_RATE,
                "passed": results.path_pass_rate >= SLO_FLOW_PATH_PASS_RATE,
            },
            "question_pass_rate": {
                "value": results.question_pass_rate,
                "target": SLO_FLOW_QUESTION_PASS_RATE,
                "passed": results.question_pass_rate >= SLO_FLOW_QUESTION_PASS_RATE,
            },
            "relevance": {
                "value": results.avg_relevance,
                "target": SLO_FLOW_RELEVANCE,
                "passed": results.avg_relevance >= SLO_FLOW_RELEVANCE,
            },
            "groundedness": {
                "value": results.avg_grounded,
                "target": SLO_FLOW_GROUNDED,
                "passed": results.avg_grounded >= SLO_FLOW_GROUNDED,
            },
        },
        "tracked_metrics": {
            "avg_latency_ms": results.avg_latency_per_question_ms,
            "p95_latency_ms": results.p95_latency_ms,
        },
        "failed_paths": [
            {
                "path_id": fp.path_id,
                "questions": fp.questions,
                "steps": [
                    {
                        "question": s.question,
                        "has_answer": s.has_answer,
                        "relevance_score": s.relevance_score,
                        "grounded_score": s.grounded_score,
                        "latency_ms": s.latency_ms,
                        "judge_explanation": s.judge_explanation,
                        "error": s.error,
                    }
                    for s in fp.steps
                ],
            }
            for fp in results.failed_paths
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[dim]Results saved to {output_path}[/dim]")


# =============================================================================
# Qdrant Access Check
# =============================================================================


def _check_qdrant_access() -> bool:
    """
    Check if Qdrant storage is accessible (not locked by another process).

    Returns:
        True if accessible, False if locked
    """
    try:
        from backend.agent.rag import get_qdrant_client

        # Use shared client (already opened by ensure_qdrant_collections)
        client = get_qdrant_client()
        # Try a simple operation
        client.get_collections()
        return True
    except Exception as e:
        if "already accessed" in str(e).lower():
            return False
        # Other errors - assume accessible but broken
        logger.warning(f"Qdrant check error: {e}")
        return True


# =============================================================================
# CLI
# =============================================================================

app = typer.Typer()

# Use absolute path for baseline (relative to backend root)
_BACKEND_ROOT = Path(__file__).parent.parent.parent.resolve()
BASELINE_PATH = _BACKEND_ROOT / "data" / "processed" / "flow_eval_baseline.json"


async def _run_eval_async(
    limit: int | None,
    verbose: bool,
    parallel: bool,
    workers: int,
    no_judge: bool,
    output: str | None,
    baseline: str | None,
    set_baseline: bool,
    debug: bool,
) -> int:
    """Async implementation of the eval runner."""
    # Check if Qdrant is accessible
    if not _check_qdrant_access():
        console.print(
            Panel(
                "[red bold]ERROR: Qdrant storage is locked by another process![/red bold]\n\n"
                "[bold]Solutions:[/bold]\n"
                "  1. Stop the backend server: Ctrl+C in the uvicorn terminal\n"
                "  2. Close any Jupyter notebooks using RAG\n"
                "  3. Run with --mock for testing without real LLM/RAG",
                border_style="red",
            )
        )
        return 1

    # Warmup: trigger model loading by a simple query
    console.print("\n[dim]Warming up models...[/dim]")
    try:
        from backend.agent.rag import tool_docs_rag

        tool_docs_rag("warmup", top_k=1)  # Trigger embedding model load
        console.print("[dim]Models loaded.[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Model preload failed: {e}[/yellow]")

    # Show tree stats first
    stats = get_tree_stats()
    console.print("\n[bold]Question Tree Stats:[/bold]")
    for key, value in stats.items():
        console.print(f"  [dim]{key}:[/dim] {value}")

    # Run evaluation
    use_judge = not no_judge
    concurrency = workers if parallel else 1
    try:
        results = await run_flow_eval(
            max_paths=limit,
            verbose=verbose,
            use_judge=use_judge,
            concurrency=concurrency,
        )
    except Exception as e:
        console.print(f"\n[red bold]ERROR: Evaluation failed: {e}[/red bold]")
        import traceback

        traceback.print_exc()
        return 1

    # Print summary
    print_summary(results)

    # Debug output for failing paths
    if debug and results.failed_paths:
        console.print("\n" + "=" * 80)
        console.print("[bold yellow]DEBUG: Full details for failed paths[/bold yellow]")
        console.print("=" * 80)

        for fp in results.failed_paths[:10]:
            console.print(f"\n[bold cyan]--- Path {fp.path_id} ---[/bold cyan]")
            for i, step in enumerate(fp.steps):
                status = "PASS" if step.passed else "FAIL"
                console.print(f"[bold]Q{i + 1}:[/bold] {step.question}")
                console.print(f"    [{status}] R={step.relevance_score} G={step.grounded_score}")
                console.print(f"    [bold]Answer:[/bold] {step.answer[:200]}...")
                if step.judge_explanation:
                    console.print(f"    [bold]Judge:[/bold] {step.judge_explanation}")
            console.print("-" * 40)

    # Save if requested
    if output:
        save_results(results, Path(output))

    # Baseline comparison
    baseline_path = Path(baseline) if baseline else BASELINE_PATH
    is_regression, baseline_score = compare_to_baseline(
        results.question_pass_rate,
        baseline_path,
        score_key="question_pass_rate",
    )
    print_baseline_comparison(results.question_pass_rate, baseline_score, is_regression)

    # Save as new baseline if requested
    if set_baseline:
        baseline_data = {
            "path_pass_rate": results.path_pass_rate,
            "question_pass_rate": results.question_pass_rate,
            "avg_relevance": results.avg_relevance,
            "avg_grounded": results.avg_grounded,
        }
        save_baseline(baseline_data, BASELINE_PATH)

    # Cleanup Qdrant client to avoid shutdown errors
    try:
        from backend.agent.rag import close_qdrant_client

        close_qdrant_client()
    except Exception:
        pass  # Ignore cleanup errors

    # Check SLOs for exit code
    slo_results = {
        "Path Pass Rate": results.path_pass_rate >= SLO_FLOW_PATH_PASS_RATE,
        "Question Pass Rate": results.question_pass_rate >= SLO_FLOW_QUESTION_PASS_RATE,
        "Relevance": results.avg_relevance >= SLO_FLOW_RELEVANCE,
        "Groundedness": results.avg_grounded >= SLO_FLOW_GROUNDED,
    }

    all_slos_passed = all(slo_results.values())
    return determine_exit_code(all_slos_passed, is_regression)


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output for each question"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel", "-p", help="Run flows in parallel"
    ),
    workers: int = typer.Option(5, "--workers", "-w", help="Max parallel workers"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge evaluation"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
    baseline: str | None = typer.Option(
        None, "--baseline", "-b", help="Path to baseline JSON for regression comparison"
    ),
    set_baseline: bool = typer.Option(
        False, "--set-baseline", help="Save current results as new baseline"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing paths"),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    exit_code = asyncio.run(
        _run_eval_async(
            limit=limit,
            verbose=verbose,
            parallel=parallel,
            workers=workers,
            no_judge=no_judge,
            output=output,
            baseline=baseline,
            set_baseline=set_baseline,
            debug=debug,
        )
    )
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
