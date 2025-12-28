"""
RAG Evaluation Harness for Acme CRM docs.

Evaluates the RAG pipeline using:
- Context relevance: Did we retrieve the right documents?
- Answer relevance: Does the answer address the question?
- Groundedness: Is the answer grounded in the context (no hallucinations)?

Uses LLM-as-judge for relevance and groundedness scoring.

Usage:
    python -m backend.rag.eval.docs_eval
    python -m backend.rag.eval.docs_eval --verbose
    python -m backend.rag.eval.docs_eval --parallel --workers 8
"""

import json
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

# RAGAS imports (optional - graceful fallback if not installed)
try:
    import nest_asyncio
    nest_asyncio.apply()
    from ragas import SingleTurnSample
    # Use pre-instantiated metrics (lowercase) - class imports are deprecated
    from ragas.metrics import faithfulness, answer_correctness
    from ragas.llms import llm_factory
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

# Preload embedding and reranker models (simulates server startup)
from backend.rag.retrieval.preload import preload_models
print("Preloading models...")
_preload_result = preload_models()
print(f"Models loaded in {_preload_result['total_ms']}ms")
print()

import typer
from rich.progress import track, Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from backend.rag.retrieval.base import create_backend
from backend.rag.pipeline.docs import answer_question
from backend.rag.eval.models import (
    EvalResult,
    DocsEvalSummary,
    SLO_CONTEXT_RELEVANCE,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
    SLO_RAG_TRIAD,
    SLO_DOC_RECALL,
    SLO_LATENCY_P95_MS,
    SLO_EVAL_LATENCY_P95_MS,
)
from backend.rag.eval.judge import judge_response, compute_doc_recall
from backend.rag.eval.base import (
    console,
    create_summary_table,
    create_detail_table,
    print_eval_header,
    print_issues_panel,
    format_check_mark,
    add_separator_row,
    compare_to_baseline,
    save_baseline,
    print_baseline_comparison,
)
from backend.rag.eval.tracking import print_full_tracking_report


# =============================================================================
# RAGAS Evaluation Functions
# =============================================================================

def get_ragas_metrics() -> tuple | None:
    """Initialize RAGAS metrics (cached).

    Uses pre-instantiated metrics from ragas.metrics.
    These use the default LLM (gpt-4o-mini via OpenAI API key from env).
    """
    if not RAGAS_AVAILABLE:
        return None
    try:
        # Pre-instantiated metrics are ready to use
        return faithfulness, answer_correctness
    except Exception:
        return None


async def evaluate_ragas_async(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None = None,
) -> tuple[float | None, float | None]:
    """Evaluate faithfulness and answer correctness using RAGAS."""
    metrics = get_ragas_metrics()
    if metrics is None:
        return None, None

    faith_metric, corr_metric = metrics
    faith_score = None
    corr_score = None

    try:
        # Faithfulness (answer grounded in context)
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        faith_score = await faith_metric.single_turn_ascore(sample)

        # Answer correctness (if reference answer available)
        if reference:
            sample_with_ref = SingleTurnSample(
                user_input=question,
                response=answer,
                reference=reference,
            )
            corr_score = await corr_metric.single_turn_ascore(sample_with_ref)
    except Exception:
        pass

    return faith_score, corr_score


def evaluate_ragas_sync(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None = None,
) -> tuple[float | None, float | None]:
    """Synchronous wrapper for RAGAS evaluation."""
    if not RAGAS_AVAILABLE:
        return None, None
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            evaluate_ragas_async(question, answer, contexts, reference)
        )
    except Exception:
        return None, None


# =============================================================================
# Configuration
# =============================================================================

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_eval_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    """Load evaluation questions from JSON file."""
    with open(path) as f:
        return json.load(f)


def evaluate_question(
    question_data: dict,
    backend,
    verbose: bool = False,
    backend_lock=None,  # threading.Lock or None
) -> EvalResult:
    """
    Evaluate a single question through the RAG pipeline.

    Args:
        question_data: Dict with id, question, target_doc_ids, expected_answer (optional)
        backend: Initialized RetrievalBackend
        verbose: Print progress
        backend_lock: Optional lock for thread-safe backend access

    Returns:
        EvalResult with all metrics including RAGAS scores
    """
    question_id = question_data["id"]
    question = question_data["question"]
    target_doc_ids = question_data["target_doc_ids"]
    category = question_data.get("category", "single_doc")
    expected_answer = question_data.get("expected_answer")  # Gold answer for RAGAS

    if verbose:
        print(f"\nEvaluating: {question_id}")
        print(f"  Question: {question[:60]}...")

    # Run RAG pipeline (with optional lock for thread safety)
    start_time = time.time()
    if backend_lock:
        with backend_lock:
            result = answer_question(question, backend, k=8, verbose=False)
    else:
        result = answer_question(question, backend, k=8, verbose=False)
    total_latency = (time.time() - start_time) * 1000

    # Build context string for judge
    context = "\n\n".join([
        f"[{c.doc_id}] {c.text[:500]}"
        for c in result["used_chunks"]
    ])

    # Extract context texts for RAGAS
    context_texts = [c.text for c in result["used_chunks"]]

    # Compute doc recall
    doc_recall = compute_doc_recall(target_doc_ids, result["doc_ids_used"])

    if verbose:
        print(f"  Retrieved: {result['doc_ids_used']}")
        print(f"  Doc recall: {doc_recall:.2f}")

    # Judge the response
    judge_result = judge_response(
        question=question,
        context=context,
        answer=result["answer"],
        doc_ids=result["doc_ids_used"],
    )

    if verbose:
        print(f"  Judge: context={judge_result.context_relevance}, "
              f"answer={judge_result.answer_relevance}, "
              f"grounded={judge_result.groundedness}")

    # RAGAS evaluation (faithfulness + answer correctness)
    ragas_faithfulness = None
    ragas_answer_correctness = None
    if RAGAS_AVAILABLE and context_texts:
        ragas_faithfulness, ragas_answer_correctness = evaluate_ragas_sync(
            question=question,
            answer=result["answer"],
            contexts=context_texts,
            reference=expected_answer,
        )
        if verbose and ragas_faithfulness is not None:
            faith_str = f"{ragas_faithfulness:.2f}"
            corr_str = f"{ragas_answer_correctness:.2f}" if ragas_answer_correctness else "N/A"
            print(f"  RAGAS: faithfulness={faith_str}, correctness={corr_str}")

    # Extract step timings from pipeline result
    step_timings = {}
    for step in result.get("steps", []):
        step_id = step.get("id", "unknown")
        elapsed = step.get("elapsed_ms", 0)
        step_timings[step_id] = elapsed

    return EvalResult(
        question_id=question_id,
        question=question,
        category=category,
        target_doc_ids=target_doc_ids,
        retrieved_doc_ids=result["doc_ids_used"],
        answer=result["answer"],
        judge_result=judge_result,
        doc_recall=doc_recall,
        latency_ms=total_latency,
        total_tokens=result["metrics"]["total_tokens"],
        step_timings=step_timings,
        max_rerank_score=result.get("max_rerank_score"),
        rerank_scores=result.get("rerank_scores", []),
        ragas_faithfulness=ragas_faithfulness,
        ragas_answer_correctness=ragas_answer_correctness,
    )


def run_evaluation(
    questions: list[dict] | None = None,
    verbose: bool = True,
    parallel: bool = False,
    max_workers: int = 4,
) -> tuple[list[EvalResult], DocsEvalSummary]:
    """
    Run full evaluation over all questions.

    Args:
        questions: List of question dicts (or load from file)
        verbose: Print progress
        parallel: Run evaluations in parallel (faster but uses more API quota)
        max_workers: Number of parallel workers (default: 4)

    Returns:
        Tuple of (results list, summary)
    """
    if questions is None:
        questions = load_eval_questions()

    mode_str = f"[cyan]parallel ({max_workers} workers)[/cyan]" if parallel else "[dim]sequential[/dim]"
    print_eval_header(
        "RAG Evaluation Harness",
        f"Evaluating [bold]{len(questions)}[/bold] questions {mode_str}",
    )

    # Initialize backend
    with console.status("[bold green]Loading backend..."):
        backend = create_backend()

    if parallel:
        results = _run_parallel(questions, backend, max_workers, verbose)
    else:
        results = _run_sequential(questions, backend, verbose)

    # Compute summary
    summary = compute_summary(results)

    return results, summary


def _run_sequential(
    questions: list[dict],
    backend,
    verbose: bool,
) -> list[EvalResult]:
    """Run evaluation sequentially."""
    results = []
    for q in track(questions, description="Evaluating..."):
        result = evaluate_question(q, backend, verbose=verbose)
        results.append(result)
    return results


def _run_parallel(
    questions: list[dict],
    backend,
    max_workers: int,
    verbose: bool,
) -> list[EvalResult]:
    """Run evaluation in parallel using ThreadPoolExecutor.

    Note: The embedding model isn't fully thread-safe, so we use a lock
    around the RAG pipeline call. The LLM judge calls still run in parallel.
    """
    total = len(questions)

    # Create a dict to preserve order
    results_by_id: dict[str, EvalResult] = {}

    # Lock for thread-safe access to the embedding model
    backend_lock = threading.Lock()

    def evaluate_with_lock(q: dict) -> EvalResult:
        """Wrapper that uses lock for backend access."""
        return evaluate_question(q, backend, False, backend_lock)

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            f"Evaluating {total} questions (max {max_workers} workers)",
            total=total,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(evaluate_with_lock, q): q
                for q in questions
            }

            # Process as they complete
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    result = future.result()
                    results_by_id[question["id"]] = result
                    progress.advance(task)
                except Exception as e:
                    progress.console.print(f"  [red]✗ {question['id']}: {e}[/red]")
                    progress.advance(task)

    # Return in original order
    results = []
    for q in questions:
        if q["id"] in results_by_id:
            results.append(results_by_id[q["id"]])

    return results


def compute_summary(results: list[EvalResult]) -> DocsEvalSummary:
    """Compute summary statistics and check SLOs.

    Quality metrics (context relevance, answer relevance, groundedness, RAG triad)
    are computed ONLY on answerable questions (excluding negative test cases).

    Negative questions are evaluated separately for correct rejection rate.
    """
    n = len(results)

    if n == 0:
        return DocsEvalSummary(
            total_tests=0,
            context_relevance=0.0,
            answer_relevance=0.0,
            groundedness=0.0,
            rag_triad_success=0.0,
            avg_doc_recall=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            total_tokens=0,
            estimated_cost=0.0,
        )

    # Separate answerable and negative questions
    answerable = [r for r in results if not r.is_negative]
    negative = [r for r in results if r.is_negative]
    n_answerable = len(answerable)
    n_negative = len(negative)

    # Compute quality metrics ONLY on answerable questions
    if n_answerable > 0:
        context_relevance = sum(r.judge_result.context_relevance for r in answerable) / n_answerable
        answer_relevance = sum(r.judge_result.answer_relevance for r in answerable) / n_answerable
        groundedness = sum(r.judge_result.groundedness for r in answerable) / n_answerable

        # RAG triad success (all three = 1)
        triad_success = sum(
            1 for r in answerable
            if r.judge_result.context_relevance == 1
            and r.judge_result.answer_relevance == 1
            and r.judge_result.groundedness == 1
        ) / n_answerable

        avg_doc_recall = sum(r.doc_recall for r in answerable) / n_answerable
    else:
        context_relevance = 0.0
        answer_relevance = 0.0
        groundedness = 0.0
        triad_success = 0.0
        avg_doc_recall = 0.0

    # Negative question handling: check if system correctly declined
    # A correct decline = answer indicates info not found/documented
    negative_handling_rate = None
    if n_negative > 0:
        decline_keywords = ["not documented", "not available", "no documentation",
                          "not found", "don't see", "don't have", "isn't documented",
                          "not mentioned", "no information"]
        correct_declines = sum(
            1 for r in negative
            if any(kw.lower() in r.answer.lower() for kw in decline_keywords)
        )
        negative_handling_rate = correct_declines / n_negative

    # Latency computed on all results
    avg_latency = sum(r.latency_ms for r in results) / n

    # P95 latency
    latencies = sorted([r.latency_ms for r in results])
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[min(p95_index, len(latencies) - 1)]

    total_tokens = sum(r.total_tokens for r in results)
    estimated_cost = (total_tokens * 0.8 * 0.40 + total_tokens * 0.2 * 1.60) / 1_000_000

    # RAGAS average scores (only for answerable results that have them)
    faithfulness_scores = [r.ragas_faithfulness for r in answerable if r.ragas_faithfulness is not None]
    correctness_scores = [r.ragas_answer_correctness for r in answerable if r.ragas_answer_correctness is not None]
    avg_ragas_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
    avg_ragas_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else None

    # Check SLOs (based on answerable questions only)
    failed_slos = []
    if context_relevance < SLO_CONTEXT_RELEVANCE:
        failed_slos.append(f"Context relevance {context_relevance:.1%} < {SLO_CONTEXT_RELEVANCE:.1%}")
    if answer_relevance < SLO_ANSWER_RELEVANCE:
        failed_slos.append(f"Answer relevance {answer_relevance:.1%} < {SLO_ANSWER_RELEVANCE:.1%}")
    if groundedness < SLO_GROUNDEDNESS:
        failed_slos.append(f"Groundedness {groundedness:.1%} < {SLO_GROUNDEDNESS:.1%}")
    if triad_success < SLO_RAG_TRIAD:
        failed_slos.append(f"RAG triad {triad_success:.1%} < {SLO_RAG_TRIAD:.1%}")
    if avg_doc_recall < SLO_DOC_RECALL:
        failed_slos.append(f"Doc recall {avg_doc_recall:.1%} < {SLO_DOC_RECALL:.1%}")
    # Use eval latency SLO (more lenient) since eval includes judge + RAGAS overhead
    if p95_latency > SLO_EVAL_LATENCY_P95_MS:
        failed_slos.append(f"P95 latency {p95_latency:.0f}ms > {SLO_EVAL_LATENCY_P95_MS}ms")

    return DocsEvalSummary(
        total_tests=n,
        answerable_tests=n_answerable,
        negative_tests=n_negative,
        context_relevance=context_relevance,
        answer_relevance=answer_relevance,
        groundedness=groundedness,
        rag_triad_success=triad_success,
        avg_doc_recall=avg_doc_recall,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
        avg_ragas_faithfulness=avg_ragas_faithfulness,
        avg_ragas_answer_correctness=avg_ragas_correctness,
        negative_handling_rate=negative_handling_rate,
        all_slos_passed=len(failed_slos) == 0,
        failed_slos=failed_slos,
    )


def print_summary(results: list[EvalResult], summary: DocsEvalSummary) -> None:
    """Print summary statistics from evaluation results using Rich."""
    n = len(results)
    
    if n == 0:
        console.print("[yellow]No results to summarize[/yellow]")
        return
    
    # Summary table using shared helper
    summary_table = create_summary_table()

    # Show test breakdown
    summary_table.add_row("Total questions", str(summary.total_tests))
    summary_table.add_row("  Answerable", f"{summary.answerable_tests} (quality metrics computed on these)")
    summary_table.add_row("  Negative tests", f"{summary.negative_tests} (correct rejection tests)")
    add_separator_row(summary_table)

    ctx_style = "[green]" if summary.context_relevance >= SLO_CONTEXT_RELEVANCE else "[red]"
    summary_table.add_row("Context Relevance", f"{ctx_style}{summary.context_relevance:.1%}[/] (SLO: ≥{SLO_CONTEXT_RELEVANCE:.0%})")
    
    ans_style = "[green]" if summary.answer_relevance >= SLO_ANSWER_RELEVANCE else "[red]"
    summary_table.add_row("Answer Relevance", f"{ans_style}{summary.answer_relevance:.1%}[/] (SLO: ≥{SLO_ANSWER_RELEVANCE:.0%})")
    
    gnd_style = "[green]" if summary.groundedness >= SLO_GROUNDEDNESS else "[red]"
    summary_table.add_row("Groundedness", f"{gnd_style}{summary.groundedness:.1%}[/] (SLO: ≥{SLO_GROUNDEDNESS:.0%})")

    # RAGAS metrics (if available)
    if summary.avg_ragas_faithfulness is not None:
        faith_style = "[green]" if summary.avg_ragas_faithfulness >= 0.7 else "[yellow]" if summary.avg_ragas_faithfulness >= 0.5 else "[red]"
        summary_table.add_row("RAGAS Faithfulness", f"{faith_style}{summary.avg_ragas_faithfulness:.1%}[/] (statement-level)")
    if summary.avg_ragas_answer_correctness is not None:
        corr_style = "[green]" if summary.avg_ragas_answer_correctness >= 0.7 else "[yellow]" if summary.avg_ragas_answer_correctness >= 0.5 else "[red]"
        summary_table.add_row("RAGAS Correctness", f"{corr_style}{summary.avg_ragas_answer_correctness:.1%}[/] (vs gold answer)")

    triad_style = "[green]" if summary.rag_triad_success >= SLO_RAG_TRIAD else "[red]"
    summary_table.add_row("RAG Triad Success", f"[bold {triad_style[1:-1]}]{summary.rag_triad_success:.1%}[/bold {triad_style[1:-1]}] (SLO: ≥{SLO_RAG_TRIAD:.0%})")
    
    recall_style = "[green]" if summary.avg_doc_recall >= SLO_DOC_RECALL else "[yellow]"
    summary_table.add_row("Avg Doc Recall", f"{recall_style}{summary.avg_doc_recall:.1%}[/] (SLO: ≥{SLO_DOC_RECALL:.0%})")
    
    add_separator_row(summary_table)
    summary_table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")

    # Use eval latency SLO (more lenient) since eval includes judge + RAGAS overhead
    p95_style = "[green]" if summary.p95_latency_ms <= SLO_EVAL_LATENCY_P95_MS else "[red]"
    summary_table.add_row("P95 Latency", f"{p95_style}{summary.p95_latency_ms:.0f}ms[/] (SLO: ≤{SLO_EVAL_LATENCY_P95_MS}ms)")
    
    summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
    summary_table.add_row("Est. Cost", f"${summary.estimated_cost:.4f}")

    # Negative question handling (if any negative tests)
    if summary.negative_handling_rate is not None:
        add_separator_row(summary_table)
        neg_style = "[green]" if summary.negative_handling_rate >= 0.9 else "[yellow]" if summary.negative_handling_rate >= 0.7 else "[red]"
        summary_table.add_row("Negative Handling", f"{neg_style}{summary.negative_handling_rate:.1%}[/] (correct rejection rate)")

    # SLO status
    add_separator_row(summary_table)
    if summary.all_slos_passed:
        summary_table.add_row("SLO Status", "[bold green]✓ ALL PASSED[/bold green]")
    else:
        summary_table.add_row("SLO Status", f"[bold red]✗ {len(summary.failed_slos)} FAILED[/bold red]")
    
    console.print(summary_table)
    
    # Per-question results using shared helper
    detail_table = create_detail_table("Per-Question Results", [
        ("ID", "left"),
        ("Ctx", "center"),
        ("Ans", "center"),
        ("Gnd", "center"),
        ("Recall", "right"),
        ("Latency", "right"),
    ])
    
    for r in results:
        detail_table.add_row(
            r.question_id,
            format_check_mark(r.judge_result.context_relevance == 1),
            format_check_mark(r.judge_result.answer_relevance == 1),
            format_check_mark(r.judge_result.groundedness == 1),
            f"{r.doc_recall:.1%}",
            f"{r.latency_ms:.0f}ms",
        )

    console.print(detail_table)

    # Rerank score analysis
    rerank_table = create_detail_table("Rerank Score Analysis", [
        ("ID", "left"),
        ("Category", "left"),
        ("Max Score", "right"),
        ("All Scores", "left"),
    ])

    for r in results:
        max_score = r.max_rerank_score if r.max_rerank_score is not None else 0.0
        # Color code: green > 0.5, yellow 0.2-0.5, red < 0.2
        if max_score >= 0.5:
            score_style = "[green]"
        elif max_score >= 0.2:
            score_style = "[yellow]"
        else:
            score_style = "[red]"

        scores_str = ", ".join(f"{s:.2f}" for s in r.rerank_scores[:5])
        if len(r.rerank_scores) > 5:
            scores_str += "..."

        rerank_table.add_row(
            r.question_id,
            r.category,
            f"{score_style}{max_score:.3f}[/]",
            scores_str,
        )

    console.print(rerank_table)

    # Summary stats for rerank scores
    answerable_scores = [r.max_rerank_score for r in results if not r.is_negative and r.max_rerank_score is not None]
    negative_scores = [r.max_rerank_score for r in results if r.is_negative and r.max_rerank_score is not None]

    if answerable_scores or negative_scores:
        console.print("\n[bold]Rerank Score Summary:[/bold]")
        if answerable_scores:
            console.print(f"  Answerable questions: min={min(answerable_scores):.3f}, max={max(answerable_scores):.3f}, avg={sum(answerable_scores)/len(answerable_scores):.3f}")
        if negative_scores:
            console.print(f"  Negative questions:   min={min(negative_scores):.3f}, max={max(negative_scores):.3f}, avg={sum(negative_scores)/len(negative_scores):.3f}")
        console.print()

    # Failed questions using shared helper
    failed = [r for r in results if r.judge_result.groundedness == 0 or r.judge_result.answer_relevance == 0]
    print_issues_panel(
        "Questions Needing Attention",
        [f"[bold]{r.question_id}[/bold]: {r.question}\n  [dim]{r.judge_result.explanation}[/dim]" for r in failed],
    )


# =============================================================================
# CLI Entrypoint
# =============================================================================

app = typer.Typer(help="RAG Evaluation Harness")


BASELINE_PATH = Path("data/processed/docs_eval_baseline.json")


@app.command()
def run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Run evaluations in parallel"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    output: Path = typer.Option(
        Path("data/processed/eval_results.json"),
        "--output", "-o",
        help="Output file for results",
    ),
    baseline: str | None = typer.Option(None, "--baseline", "-b", help="Path to baseline JSON for regression comparison"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Save current results as new baseline"),
) -> None:
    """Run evaluation on all test questions."""
    results, summary = run_evaluation(
        verbose=verbose,
        parallel=parallel,
        max_workers=workers,
    )
    print_summary(results, summary)

    # Per-question detail table
    detail_table = create_detail_table("Per-Question Results", [
        ("ID", "left"),
        ("Ctx", "center"),
        ("Ans", "center"),
        ("Gnd", "center"),
        ("Recall", "right"),
        ("Latency", "right"),
    ])

    for r in results:
        detail_table.add_row(
            r.question_id,
            format_check_mark(r.judge_result.context_relevance == 1),
            format_check_mark(r.judge_result.answer_relevance == 1),
            format_check_mark(r.judge_result.groundedness == 1),
            f"{r.doc_recall:.1%}",
            f"{r.latency_ms:.0f}ms",
        )

    console.print(detail_table)

    # Failed questions
    failed = [r for r in results if r.judge_result.groundedness == 0 or r.judge_result.answer_relevance == 0]
    print_issues_panel(
        "Questions Needing Attention",
        [f"[bold]{r.question_id}[/bold]: {r.question}\n  [dim]{r.judge_result.explanation}[/dim]" for r in failed],
    )

    # Save results to file
    output.parent.mkdir(parents=True, exist_ok=True)
    results_data = {
        "results": [r.model_dump() for r in results],
        "summary": summary.model_dump(),
    }
    with open(output, "w") as f:
        json.dump(results_data, f, indent=2)

    console.print(f"\n[dim]Results saved to {output}[/dim]")

    # Print tracking report (regression detection + budget analysis)
    print_full_tracking_report(results, summary)

    # Baseline comparison
    baseline_path = Path(baseline) if baseline else BASELINE_PATH
    is_regression, baseline_score = compare_to_baseline(
        summary.rag_triad_success,
        baseline_path,
        score_key="rag_triad_success",
    )
    print_baseline_comparison(summary.rag_triad_success, baseline_score, is_regression)

    # Save as new baseline if requested
    if set_baseline:
        save_baseline(summary.model_dump(), BASELINE_PATH)

    # Exit code based on SLOs and regression
    exit_code = 0

    if not summary.all_slos_passed:
        console.print("\n[red bold]FAIL: One or more SLOs not met[/red bold]")
        for slo in summary.failed_slos:
            console.print(f"  • {slo}")
        exit_code = 1

    if is_regression:
        console.print("\n[red bold]FAIL: Regression detected[/red bold]")
        exit_code = 1

    if exit_code == 0:
        console.print("\n[green bold]✓ PASS: All SLOs met[/green bold]")

    raise typer.Exit(code=exit_code)


def main() -> None:
    """Main entrypoint for evaluation."""
    app()


if __name__ == "__main__":
    main()
