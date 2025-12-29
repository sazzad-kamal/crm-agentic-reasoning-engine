"""
Conversation flow evaluation - tests all paths through the question tree.

This eval simulates a user clicking through the demo, testing each possible
path with accumulating conversation memory.

Usage:
    python -m backend.agent.eval.flow_eval
    python -m backend.agent.eval.flow_eval --max-paths 10  # Quick test
    python -m backend.agent.eval.flow_eval --verbose  # Show all responses
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file before any other imports
from dotenv import load_dotenv
load_dotenv()

from rich.table import Table
from rich.panel import Panel

from backend.agent.question_tree import generate_all_paths, get_tree_stats
from backend.agent.eval.base import (
    console,
    format_percentage,
    format_check_mark,
    print_eval_header,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SLO Thresholds for Flow Evaluation
# =============================================================================

# Quality SLOs
SLO_PATH_PASS_RATE = 0.85       # 85% of conversation paths should pass
SLO_QUESTION_PASS_RATE = 0.90  # 90% of individual questions should pass
SLO_RELEVANCE = 0.85           # 85% relevance score
SLO_GROUNDED = 0.80            # 80% groundedness score

# Latency SLOs
SLO_AVG_LATENCY_MS = 4000      # 4s average per question
SLO_P95_LATENCY_MS = 8000      # 8s P95 per question (flow has multi-turn overhead)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class StepResult:
    """Result of a single question in a flow."""
    question: str
    answer: str
    latency_ms: int
    has_answer: bool
    has_sources: bool
    # LLM Judge scores
    relevance_score: int = 0  # 0 or 1
    grounded_score: int = 0   # 0 or 1
    judge_explanation: str = ""
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Question passes if has answer AND both judge scores are 1."""
        return self.has_answer and self.relevance_score == 1 and self.grounded_score == 1


@dataclass
class FlowResult:
    """Result of testing a complete conversation flow."""
    path_id: int
    questions: list[str]
    steps: list[StepResult]
    total_latency_ms: int
    success: bool
    error: str | None = None


@dataclass
class EvalResults:
    """Aggregated results from all flow tests."""
    total_paths: int
    paths_tested: int
    paths_passed: int
    paths_failed: int
    total_questions: int
    questions_passed: int
    questions_failed: int
    # Judge metrics
    avg_relevance: float = 0.0
    avg_grounded: float = 0.0
    total_latency_ms: int = 0
    avg_latency_per_question_ms: float = 0.0
    p95_latency_ms: float = 0.0  # P95 latency per question
    wall_clock_ms: int = 0  # Total wall-clock time for the eval
    failed_paths: list[FlowResult] = field(default_factory=list)
    all_results: list[FlowResult] = field(default_factory=list)

    @property
    def path_pass_rate(self) -> float:
        """Percentage of paths that passed."""
        return self.paths_passed / self.paths_tested if self.paths_tested > 0 else 0.0

    @property
    def question_pass_rate(self) -> float:
        """Percentage of questions that passed."""
        return self.questions_passed / self.total_questions if self.total_questions > 0 else 0.0


# =============================================================================
# LLM Judge (reuse from e2e_eval)
# =============================================================================

FLOW_JUDGE_SYSTEM = """You are an expert evaluator for a CRM assistant conversation flow.
Evaluate the quality of each answer in a multi-turn conversation.

Score each criterion as 0 or 1:

1. ANSWER_RELEVANCE: Does the answer address the user's question?
   - 1 if the answer directly addresses what was asked
   - 0 if the answer is off-topic, too generic, or doesn't address the question

2. ANSWER_GROUNDED: Is the answer based on real CRM data?
   - 1 if mentions specific companies, dates, values, numbers, or contact names
   - 1 if appropriately says data is not available (honest grounding)
   - 0 if the answer seems made up, hallucinates facts, or is vague ("several", "some")

Respond in JSON:
{
  "answer_relevance": 0 or 1,
  "answer_grounded": 0 or 1,
  "explanation": "brief explanation"
}"""

FLOW_JUDGE_PROMPT = """Question: {question}

Conversation context: {context}

Answer: {answer}

Evaluate this response:"""


def judge_answer(
    question: str,
    answer: str,
    context: str = "",
) -> dict:
    """Judge an answer using LLM."""
    from backend.common.llm_client import call_llm

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

        import json
        result = json.loads(response)
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
) -> StepResult:
    """
    Test a single question with conversation history.

    Args:
        question: The question to ask
        history: List of {question, answer} dicts for memory
        session_id: Session ID for the conversation
        use_judge: Whether to run LLM-as-judge evaluation

    Returns:
        StepResult with answer and metrics
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
            context = "\n".join([
                f"Q: {h['question']}\nA: {h['answer'][:200]}..."
                for h in history[-2:]  # Last 2 turns
            ])

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

        return StepResult(
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
        return StepResult(
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
    steps: list[StepResult] = []
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
        history.append({
            "question": question,
            "answer": step_result.answer,
        })

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
) -> EvalResults:
    """
    Run the flow evaluation on all paths.

    Args:
        max_paths: Limit number of paths to test (None = all)
        verbose: Print detailed output
        use_judge: Whether to run LLM-as-judge evaluation
        concurrency: Number of flows to run in parallel (default 5)

    Returns:
        EvalResults with aggregated metrics
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
                console.print(f"[dim][{completed}/{total}][/dim] Path {path_id+1}: [{status_color}]{status}[/{status_color}] ({result.total_latency_ms}ms)")
                if verbose or not result.success:
                    for j, step in enumerate(result.steps):
                        step_color = "green" if step.passed else "red"
                        status_icon = "PASS" if step.passed else "FAIL"
                        console.print(f"  Q{j+1}: {step.question[:50]}...")
                        console.print(f"      [{step_color}][{status_icon}][/{step_color}] R={step.relevance_score} G={step.grounded_score} | {step.latency_ms}ms")
                        if step.judge_explanation and verbose:
                            console.print(f"      [dim]Judge: {step.judge_explanation[:80]}...[/dim]")
                        if step.error:
                            console.print(f"      [red]ERROR: {step.error}[/red]")
            return result

    # Run all flows in parallel (limited by semaphore)
    tasks = [
        run_with_semaphore(path, i)
        for i, path in enumerate(paths_to_test)
    ]
    console.print(f"[cyan]Starting {len(tasks)} flows...[/cyan]")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for exceptions in results
    actual_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            console.print(f"[red]Path {i+1} raised exception: {r}[/red]")
            # Create a failed result
            actual_results.append(FlowResult(
                path_id=i,
                questions=paths_to_test[i],
                steps=[],
                total_latency_ms=0,
                success=False,
                error=str(r),
            ))
        else:
            actual_results.append(r)
    results = actual_results

    # Aggregate results
    paths_passed = sum(1 for r in results if r.success)
    paths_failed = len(results) - paths_passed

    total_questions = sum(len(r.steps) for r in results)
    questions_passed = sum(
        sum(1 for s in r.steps if s.passed) for r in results
    )
    questions_failed = total_questions - questions_passed

    # Calculate judge score averages
    all_steps = [s for r in results for s in r.steps]
    avg_relevance = sum(s.relevance_score for s in all_steps) / len(all_steps) if all_steps else 0
    avg_grounded = sum(s.grounded_score for s in all_steps) / len(all_steps) if all_steps else 0

    total_latency = sum(r.total_latency_ms for r in results)
    avg_latency = total_latency / total_questions if total_questions > 0 else 0

    # Calculate P95 latency per question
    step_latencies = sorted([s.latency_ms for s in all_steps])
    p95_idx = int(len(step_latencies) * 0.95) if step_latencies else 0
    p95_latency = step_latencies[min(p95_idx, len(step_latencies) - 1)] if step_latencies else 0.0

    failed_paths = [r for r in results if not r.success]
    wall_clock_ms = int((time.time() - eval_start_time) * 1000)

    eval_results = EvalResults(
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


def print_summary(results: EvalResults):
    """Print a comprehensive summary of eval results with SLO status."""
    console.print()

    # ==========================================================================
    # Main Summary Table
    # ==========================================================================
    summary_table = Table(title="Flow Evaluation Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")
    summary_table.add_column("SLO", justify="right", style="dim")
    summary_table.add_column("Status", justify="center")

    # Path metrics
    path_pass_rate = results.path_pass_rate
    path_slo_pass = path_pass_rate >= SLO_PATH_PASS_RATE
    summary_table.add_row(
        "Paths Tested",
        f"{results.paths_tested}/{results.total_paths}",
        "",
        "",
    )
    summary_table.add_row(
        "Path Pass Rate",
        format_percentage(path_pass_rate),
        f"≥{format_percentage(SLO_PATH_PASS_RATE)}",
        format_check_mark(path_slo_pass),
    )
    summary_table.add_row("", "", "", "")  # Spacer

    # Question metrics
    q_pass_rate = results.question_pass_rate
    q_slo_pass = q_pass_rate >= SLO_QUESTION_PASS_RATE
    summary_table.add_row(
        "Questions Total",
        str(results.total_questions),
        "",
        "",
    )
    summary_table.add_row(
        "Question Pass Rate",
        format_percentage(q_pass_rate),
        f"≥{format_percentage(SLO_QUESTION_PASS_RATE)}",
        format_check_mark(q_slo_pass),
    )
    summary_table.add_row("", "", "", "")  # Spacer

    # Judge score metrics
    relevance_slo_pass = results.avg_relevance >= SLO_RELEVANCE
    grounded_slo_pass = results.avg_grounded >= SLO_GROUNDED
    summary_table.add_row(
        "[bold]LLM Judge Scores[/bold]",
        "",
        "",
        "",
    )
    summary_table.add_row(
        "  Relevance",
        format_percentage(results.avg_relevance),
        f"≥{format_percentage(SLO_RELEVANCE)}",
        format_check_mark(relevance_slo_pass),
    )
    summary_table.add_row(
        "  Groundedness",
        format_percentage(results.avg_grounded),
        f"≥{format_percentage(SLO_GROUNDED)}",
        format_check_mark(grounded_slo_pass),
    )
    summary_table.add_row("", "", "", "")  # Spacer

    # Latency metrics
    avg_latency_slo_pass = results.avg_latency_per_question_ms <= SLO_AVG_LATENCY_MS
    p95_latency_slo_pass = results.p95_latency_ms <= SLO_P95_LATENCY_MS
    summary_table.add_row(
        "[bold]Latency[/bold]",
        "",
        "",
        "",
    )
    summary_table.add_row(
        "  Avg per Question",
        f"{results.avg_latency_per_question_ms:.0f}ms",
        f"≤{SLO_AVG_LATENCY_MS}ms",
        format_check_mark(avg_latency_slo_pass),
    )
    summary_table.add_row(
        "  P95 per Question",
        f"{results.p95_latency_ms:.0f}ms",
        f"≤{SLO_P95_LATENCY_MS}ms",
        format_check_mark(p95_latency_slo_pass),
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
        f"{wall_secs:.1f}s ({wall_secs/60:.1f} min)",
        "",
        "",
    )

    console.print(summary_table)

    # ==========================================================================
    # SLO Summary Panel
    # ==========================================================================
    slo_checks = [
        ("Path Pass Rate", path_slo_pass, format_percentage(path_pass_rate), f"≥{format_percentage(SLO_PATH_PASS_RATE)}"),
        ("Question Pass Rate", q_slo_pass, format_percentage(q_pass_rate), f"≥{format_percentage(SLO_QUESTION_PASS_RATE)}"),
        ("Relevance", relevance_slo_pass, format_percentage(results.avg_relevance), f"≥{format_percentage(SLO_RELEVANCE)}"),
        ("Groundedness", grounded_slo_pass, format_percentage(results.avg_grounded), f"≥{format_percentage(SLO_GROUNDED)}"),
        ("Avg Latency", avg_latency_slo_pass, f"{results.avg_latency_per_question_ms:.0f}ms", f"≤{SLO_AVG_LATENCY_MS}ms"),
        ("P95 Latency", p95_latency_slo_pass, f"{results.p95_latency_ms:.0f}ms", f"≤{SLO_P95_LATENCY_MS}ms"),
    ]

    passed_slos = sum(1 for _, passed, _, _ in slo_checks if passed)
    total_slos = len(slo_checks)
    all_slos_passed = passed_slos == total_slos

    console.print()
    slo_table = Table(title="SLO Summary", show_header=True, header_style="bold")
    slo_table.add_column("SLO", style="bold")
    slo_table.add_column("Actual", justify="right")
    slo_table.add_column("Target", justify="right", style="dim")
    slo_table.add_column("Status", justify="center")

    for name, passed, actual, target in slo_checks:
        slo_table.add_row(name, actual, target, format_check_mark(passed))

    console.print(slo_table)

    # Failed SLOs detail
    failed_slos = [name for name, passed, _, _ in slo_checks if not passed]
    if failed_slos:
        console.print(f"\n[red bold][!] {len(failed_slos)} SLO(s) FAILED:[/red bold]")
        for slo_name in failed_slos:
            console.print(f"    [red]✗[/red] {slo_name}")
    else:
        console.print(f"\n[green bold][OK] All {total_slos} SLOs passed[/green bold]")

    # ==========================================================================
    # Failed Paths Detail
    # ==========================================================================
    if results.failed_paths:
        console.print()
        failed_table = Table(title=f"Failed Paths ({len(results.failed_paths)} total, showing first 5)", show_header=True, header_style="bold yellow")
        failed_table.add_column("Path", style="bold", width=6)
        failed_table.add_column("Question", width=50)
        failed_table.add_column("R", justify="center", width=3)
        failed_table.add_column("G", justify="center", width=3)
        failed_table.add_column("Latency", justify="right", width=8)
        failed_table.add_column("Issue", width=40)

        for fp in results.failed_paths[:5]:
            for i, step in enumerate(fp.steps):
                if not step.passed:
                    issue = step.judge_explanation[:40] if step.judge_explanation else (step.error or "Unknown")
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
    console.print()
    if all_slos_passed and results.paths_failed == 0:
        console.print(Panel(
            "[green bold]OVERALL: PASS[/green bold]\n"
            f"All {results.paths_tested} paths passed, all SLOs met",
            border_style="green",
        ))
    else:
        failure_reasons = []
        if results.paths_failed > 0:
            failure_reasons.append(f"{results.paths_failed} paths failed")
        if failed_slos:
            failure_reasons.append(f"{len(failed_slos)} SLOs not met: {', '.join(failed_slos)}")
        console.print(Panel(
            "[red bold]OVERALL: FAIL[/red bold]\n"
            f"{'; '.join(failure_reasons)}",
            border_style="red",
        ))


def save_results(results: EvalResults, output_path: Path):
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
            "path_pass_rate": {"value": results.path_pass_rate, "target": SLO_PATH_PASS_RATE, "passed": results.path_pass_rate >= SLO_PATH_PASS_RATE},
            "question_pass_rate": {"value": results.question_pass_rate, "target": SLO_QUESTION_PASS_RATE, "passed": results.question_pass_rate >= SLO_QUESTION_PASS_RATE},
            "relevance": {"value": results.avg_relevance, "target": SLO_RELEVANCE, "passed": results.avg_relevance >= SLO_RELEVANCE},
            "groundedness": {"value": results.avg_grounded, "target": SLO_GROUNDED, "passed": results.avg_grounded >= SLO_GROUNDED},
            "avg_latency_ms": {"value": results.avg_latency_per_question_ms, "target": SLO_AVG_LATENCY_MS, "passed": results.avg_latency_per_question_ms <= SLO_AVG_LATENCY_MS},
            "p95_latency_ms": {"value": results.p95_latency_ms, "target": SLO_P95_LATENCY_MS, "passed": results.p95_latency_ms <= SLO_P95_LATENCY_MS},
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
        from backend.rag.retrieval.constants import QDRANT_PATH
        from qdrant_client import QdrantClient

        # Try to open the Qdrant storage
        client = QdrantClient(path=str(QDRANT_PATH))
        # Try a simple operation
        client.get_collections()
        client.close()
        return True
    except Exception as e:
        if "already accessed" in str(e).lower():
            return False
        # Other errors - assume accessible but broken
        logger.warning(f"Qdrant check error: {e}")
        return True


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Run conversation flow evaluation")
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Maximum number of paths to test (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each question",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no LLM calls, uses canned responses)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (faster, but no quality scores)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of flows to run in parallel (default: 5)",
    )

    args = parser.parse_args()

    # Enable mock mode if requested
    if args.mock:
        import os
        os.environ["MOCK_LLM"] = "true"
        console.print("\n[yellow][MOCK MODE] Using canned responses, no LLM calls[/yellow]")
    else:
        # Check if Qdrant is accessible (only when not in mock mode)
        if not _check_qdrant_access():
            console.print(Panel(
                "[red bold]ERROR: Qdrant storage is locked by another process![/red bold]\n\n"
                "[bold]Solutions:[/bold]\n"
                "  1. Stop the backend server: Ctrl+C in the uvicorn terminal\n"
                "  2. Close any Jupyter notebooks using RAG\n"
                "  3. Run with --mock for testing without real LLM/RAG",
                border_style="red",
            ))
            return 1

        # Warmup: preload models before parallel execution
        console.print("\n[dim]Warming up models...[/dim]")
        try:
            from backend.rag.retrieval.preload import preload_models
            preload_models(parallel=False)  # Sequential to avoid race
            console.print("[dim]Models loaded.[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Model preload failed: {e}[/yellow]")

    # Show tree stats first
    stats = get_tree_stats()
    console.print("\n[bold]Question Tree Stats:[/bold]")
    for key, value in stats.items():
        console.print(f"  [dim]{key}:[/dim] {value}")

    # Run evaluation
    use_judge = not args.no_judge
    try:
        results = await run_flow_eval(
            max_paths=args.max_paths,
            verbose=args.verbose,
            use_judge=use_judge,
            concurrency=args.concurrency,
        )
    except Exception as e:
        console.print(f"\n[red bold]ERROR: Evaluation failed: {e}[/red bold]")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print_summary(results)

    # Save if requested
    if args.output:
        save_results(results, Path(args.output))

    # Cleanup Qdrant clients to avoid shutdown errors
    try:
        from backend.rag.retrieval import clear_backend_cache, clear_private_backend_cache
        from backend.rag.retrieval.base import _backend_cache
        from backend.rag.retrieval.private import _private_backend_cache

        # Close the actual Qdrant clients if they exist
        if _backend_cache is not None and hasattr(_backend_cache, 'client'):
            try:
                _backend_cache.client.close()
            except Exception:
                pass
        if _private_backend_cache is not None and hasattr(_private_backend_cache, 'client'):
            try:
                _private_backend_cache.client.close()
            except Exception:
                pass

        clear_backend_cache()
        clear_private_backend_cache()
    except Exception:
        pass  # Ignore cleanup errors

    # Exit with error code if failures
    return 0 if results.paths_failed == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    exit_code = asyncio.run(main())
    exit(exit_code)
