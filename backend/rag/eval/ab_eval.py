"""
A/B Evaluation for RAG Pipeline Configuration.

Runs the same evaluation questions with different pipeline configurations
to quantify the quality vs latency tradeoffs of HyDE, query rewrite, and reranking.

Usage:
    python -m backend.rag.eval.ab_eval
    python -m backend.rag.eval.ab_eval --limit 5
    python -m backend.rag.eval.ab_eval --parallel --workers 8
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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
from rich.table import Table
from rich.progress import track

from backend.rag.retrieval.base import create_backend
from backend.rag.pipeline.docs import answer_question
from backend.rag.eval.docs_eval import load_eval_questions
from backend.rag.eval.judge import judge_response
from backend.rag.eval.base import (
    console,
    format_check_mark,
    format_percentage,
    format_delta,
    print_eval_header,
)
app = typer.Typer(help="A/B Evaluation for pipeline configurations")


# =============================================================================
# Configuration Matrix (2x2x2 = 8 configs)
# =============================================================================

AB_CONFIGS: dict[str, dict[str, bool]] = {
    # All on (quality)
    "full_quality": {"use_hyde": True, "use_rewrite": True, "use_reranker": True},

    # Single feature off
    "no_reranker": {"use_hyde": True, "use_rewrite": True, "use_reranker": False},
    "no_hyde": {"use_hyde": False, "use_rewrite": True, "use_reranker": True},
    "no_rewrite": {"use_hyde": True, "use_rewrite": False, "use_reranker": True},

    # Two features off
    "hyde_only": {"use_hyde": True, "use_rewrite": False, "use_reranker": False},
    "rewrite_only": {"use_hyde": False, "use_rewrite": True, "use_reranker": False},
    "reranker_only": {"use_hyde": False, "use_rewrite": False, "use_reranker": True},

    # All off (fast)
    "fast": {"use_hyde": False, "use_rewrite": False, "use_reranker": False},
}


# =============================================================================
# A/B Evaluation
# =============================================================================

def _evaluate_single_question(
    q: dict,
    config: dict[str, bool],
    backend,
    backend_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    """Evaluate a single question with the given config.

    Args:
        q: Question dict with question and target_doc_ids
        config: Config dict with use_hyde, use_rewrite, use_reranker
        backend: Initialized RetrievalBackend
        backend_lock: Optional lock for thread-safe backend access

    Returns:
        Dict with question-level metrics
    """
    question = q["question"]
    target_doc_ids = q["target_doc_ids"]

    start_time = time.time()

    try:
        # Run RAG (with optional lock for thread safety)
        if backend_lock:
            with backend_lock:
                result = answer_question(
                    question,
                    backend,
                    k=8,
                    use_hyde=config["use_hyde"],
                    use_rewrite=config["use_rewrite"],
                    verbose=False,
                )
        else:
            result = answer_question(
                question,
                backend,
                k=8,
                use_hyde=config["use_hyde"],
                use_rewrite=config["use_rewrite"],
                verbose=False,
            )
        latency = (time.time() - start_time) * 1000

        # Build context for judge
        context = "\n\n".join([
            f"[{c.doc_id}] {c.text[:500]}"
            for c in result["used_chunks"]
        ])

        # Judge the response
        judge_result = judge_response(
            question=question,
            context=context,
            answer=result["answer"],
            doc_ids=result["doc_ids_used"],
        )

        # Compute doc recall
        target_set = set(target_doc_ids)
        retrieved_set = set(result["doc_ids_used"])
        doc_recall = len(target_set & retrieved_set) / len(target_set) if target_set else 1.0

        return {
            "id": q["id"],
            "latency_ms": latency,
            "context_relevance": judge_result.context_relevance,
            "answer_relevance": judge_result.answer_relevance,
            "groundedness": judge_result.groundedness,
            "doc_recall": doc_recall,
            "rag_triad": int(
                judge_result.context_relevance == 1 and
                judge_result.answer_relevance == 1 and
                judge_result.groundedness == 1
            ),
        }

    except Exception as e:
        return {
            "id": q["id"],
            "latency_ms": (time.time() - start_time) * 1000,
            "context_relevance": 0,
            "answer_relevance": 0,
            "groundedness": 0,
            "doc_recall": 0,
            "rag_triad": 0,
            "error": str(e),
        }


def run_config_eval(
    config_name: str,
    config: dict[str, bool],
    questions: list[dict],
    backend,
    parallel: bool = False,
    max_workers: int = 4,
) -> dict[str, Any]:
    """
    Run evaluation with a specific configuration.

    Args:
        config_name: Name of the config
        config: Config dict with use_hyde, use_rewrite, use_reranker
        questions: List of question dicts
        backend: Initialized RetrievalBackend
        parallel: Run evaluations in parallel
        max_workers: Number of parallel workers

    Returns:
        Dict with metrics for this configuration
    """
    if parallel:
        results = _run_config_parallel(config, questions, backend, max_workers)
    else:
        results = _run_config_sequential(config, questions, backend)

    # Aggregate metrics
    n = len(results)
    latencies = sorted([r["latency_ms"] for r in results])
    p95_idx = int(n * 0.95) if n > 0 else 0

    return {
        "config_name": config_name,
        "use_hyde": config["use_hyde"],
        "use_rewrite": config["use_rewrite"],
        "use_reranker": config["use_reranker"],
        "num_questions": n,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / n if n else 0,
        "p95_latency_ms": latencies[min(p95_idx, n - 1)] if n else 0,
        "context_relevance": sum(r["context_relevance"] for r in results) / n if n else 0,
        "answer_relevance": sum(r["answer_relevance"] for r in results) / n if n else 0,
        "groundedness": sum(r["groundedness"] for r in results) / n if n else 0,
        "rag_triad": sum(r["rag_triad"] for r in results) / n if n else 0,
        "doc_recall": sum(r["doc_recall"] for r in results) / n if n else 0,
    }


def _run_config_sequential(
    config: dict[str, bool],
    questions: list[dict],
    backend,
) -> list[dict[str, Any]]:
    """Run config evaluation sequentially."""
    results = []
    for q in questions:
        result = _evaluate_single_question(q, config, backend)
        if "error" in result:
            console.print(f"[red]Error on question: {result['error']}[/red]")
        results.append(result)
    return results


def _run_config_parallel(
    config: dict[str, bool],
    questions: list[dict],
    backend,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Run config evaluation in parallel using ThreadPoolExecutor.

    Note: The embedding model isn't fully thread-safe, so we use a lock
    around the RAG pipeline call. The LLM judge calls still run in parallel.
    """
    results_by_id: dict[str, dict[str, Any]] = {}

    # Lock for thread-safe access to the embedding model
    backend_lock = threading.Lock()

    def evaluate_with_lock(q: dict) -> dict[str, Any]:
        """Wrapper that uses lock for backend access."""
        return _evaluate_single_question(q, config, backend, backend_lock)

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
                if "error" in result:
                    console.print(f"[red]Error on question: {result['error']}[/red]")
            except Exception as e:
                console.print(f"[red]✗ {question['id']}: {e}[/red]")
                results_by_id[question["id"]] = {
                    "id": question["id"],
                    "latency_ms": 0,
                    "context_relevance": 0,
                    "answer_relevance": 0,
                    "groundedness": 0,
                    "doc_recall": 0,
                    "rag_triad": 0,
                }

    # Return in original order
    results = []
    for q in questions:
        if q["id"] in results_by_id:
            results.append(results_by_id[q["id"]])

    return results


def run_ab_evaluation(
    limit: int | None = None,
    configs: list[str] | None = None,
    parallel: bool = False,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """
    Run A/B evaluation across all configurations.

    Args:
        limit: Limit number of questions per config
        configs: Specific configs to run (default: all)
        parallel: Run question evaluations in parallel within each config
        max_workers: Number of parallel workers

    Returns:
        List of results per configuration
    """
    mode_str = f"[cyan]parallel ({max_workers} workers)[/cyan]" if parallel else "[dim]sequential[/dim]"
    print_eval_header(
        "[bold blue]RAG A/B Evaluation[/bold blue]",
        f"Running A/B comparison across pipeline configurations {mode_str}",
    )

    # Load questions
    questions = load_eval_questions()
    if limit:
        questions = questions[:limit]

    console.print(f"Questions per config: {len(questions)}")

    # Initialize backend once
    with console.status("[bold green]Loading backend..."):
        backend = create_backend()

    # Select configs
    configs_to_run = configs or list(AB_CONFIGS.keys())

    all_results = []

    for config_name in track(configs_to_run, description="Running configs..."):
        config = AB_CONFIGS[config_name]
        console.print(f"\n[cyan]Config: {config_name}[/cyan]")
        console.print(f"  HyDE={config['use_hyde']}, Rewrite={config['use_rewrite']}, Reranker={config['use_reranker']}")

        result = run_config_eval(
            config_name, config, questions, backend,
            parallel=parallel, max_workers=max_workers,
        )
        all_results.append(result)

        console.print(f"  RAG Triad: {result['rag_triad']:.1%}, Latency: {result['avg_latency_ms']:.0f}ms")

    return all_results


def print_ab_results(results: list[dict[str, Any]]) -> None:
    """Print A/B comparison results."""
    # Sort by RAG triad (quality)
    results_sorted = sorted(results, key=lambda x: x["rag_triad"], reverse=True)

    # Main comparison table
    table = Table(title="A/B Configuration Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Config", style="bold")
    table.add_column("HyDE", justify="center")
    table.add_column("Rewrite", justify="center")
    table.add_column("Rerank", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("RAG Triad", justify="right")
    table.add_column("Recall", justify="right")

    for r in results_sorted:
        table.add_row(
            r["config_name"],
            format_check_mark(r["use_hyde"]),
            format_check_mark(r["use_rewrite"]),
            format_check_mark(r["use_reranker"]),
            f"{r['avg_latency_ms']:.0f}ms",
            format_percentage(r["rag_triad"], (0.8, 0.6)),
            format_percentage(r["doc_recall"], (0.7, 0.5)),
        )

    console.print(table)

    # Feature impact analysis
    console.print("\n[bold]Feature Impact Analysis[/bold]")

    # Find baseline (full_quality)
    baseline = next((r for r in results if r["config_name"] == "full_quality"), results[0])

    impact_table = Table(show_header=True, header_style="bold")
    impact_table.add_column("Feature Disabled")
    impact_table.add_column("Quality Impact", justify="right")
    impact_table.add_column("Latency Saved", justify="right")
    impact_table.add_column("Efficiency", justify="right")

    for r in results:
        if r["config_name"] in ["no_hyde", "no_rewrite", "no_reranker"]:
            quality_delta = r["rag_triad"] - baseline["rag_triad"]
            latency_delta = baseline["avg_latency_ms"] - r["avg_latency_ms"]

            # Efficiency = latency saved per % quality lost
            efficiency = latency_delta / abs(quality_delta * 100) if quality_delta != 0 else 0

            feature = r["config_name"].replace("no_", "").upper()

            impact_table.add_row(
                feature,
                format_delta(quality_delta, is_positive_good=True),
                f"[green]{latency_delta:+.0f}ms[/green]" if latency_delta > 0 else f"{latency_delta:.0f}ms",
                f"{efficiency:.1f}ms/%",
            )

    console.print(impact_table)

    # Recommendation
    console.print("\n[bold]Recommendation[/bold]")

    # Find most efficient feature to disable
    feature_impacts = []
    for r in results:
        if r["config_name"] in ["no_hyde", "no_rewrite", "no_reranker"]:
            quality_delta = abs(baseline["rag_triad"] - r["rag_triad"])
            latency_delta = baseline["avg_latency_ms"] - r["avg_latency_ms"]
            if quality_delta > 0 and latency_delta > 0:
                efficiency = latency_delta / quality_delta
                feature_impacts.append((r["config_name"].replace("no_", ""), efficiency, quality_delta, latency_delta))

    if feature_impacts:
        # Sort by efficiency (latency saved per quality lost)
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        best = feature_impacts[0]
        console.print(f"If you need to reduce latency, disable [bold]{best[0].upper()}[/bold] first:")
        console.print(f"  • Saves {best[3]:.0f}ms latency")
        console.print(f"  • Costs {best[2]:.1%} quality")


def save_ab_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save A/B results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


# =============================================================================
# CLI
# =============================================================================

@app.command()
def run(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit questions per config"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Run evaluations in parallel"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    output: Path = typer.Option(
        Path("backend/data/processed/ab_eval_results.json"),
        "--output", "-o",
        help="Output file for results",
    ),
    configs: str | None = typer.Option(
        None,
        "--configs", "-c",
        help="Comma-separated configs to run (default: all)",
    ),
) -> None:
    """Run A/B evaluation across pipeline configurations."""
    config_list = configs.split(",") if configs else None

    results = run_ab_evaluation(
        limit=limit,
        configs=config_list,
        parallel=parallel,
        max_workers=workers,
    )
    print_ab_results(results)
    save_ab_results(results, output)


@app.command()
def quick() -> None:
    """Quick A/B test with just 3 questions and 4 main configs."""
    results = run_ab_evaluation(
        limit=3,
        configs=["full_quality", "no_reranker", "no_hyde", "fast"],
    )
    print_ab_results(results)


if __name__ == "__main__":
    app()
