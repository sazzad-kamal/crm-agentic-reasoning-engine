"""CLI for evaluation."""

from __future__ import annotations

import asyncio
import atexit
import logging
import platform
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.panel import Panel

# Fix Windows asyncio cleanup issues with httpx/RAGAS
# Use SelectorEventLoop instead of ProactorEventLoop to avoid "Event loop is closed" errors
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment before other imports
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

from backend.agent.followup.tree import get_tree_stats
from backend.agent.rag.client import close_qdrant_client
from backend.eval.formatting import console, print_debug_failures
from backend.eval.langsmith import get_latency_percentages
from backend.eval.output import check_qdrant_access, print_summary, save_results
from backend.eval.runner import run_flow_eval

# Register cleanup to prevent shutdown errors
atexit.register(close_qdrant_client)

app = typer.Typer()


def ensure_qdrant_collections() -> None:
    """Ensure Qdrant collections exist, ingesting data if needed."""
    from backend.agent.rag.client import close_qdrant_client, get_qdrant_client
    from backend.agent.rag.config import PRIVATE_COLLECTION, QDRANT_PATH
    from backend.agent.rag.ingest import ingest_private_texts

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant = get_qdrant_client()

    private_exists = (
        qdrant.collection_exists(PRIVATE_COLLECTION)
        and (qdrant.get_collection(PRIVATE_COLLECTION).points_count or 0) > 0
    )

    if private_exists:
        print("Qdrant collections ready.")
        return

    # Close singleton before ingest (ingest needs exclusive access to local storage)
    close_qdrant_client()

    print("Ingesting private texts into Qdrant...")
    ingest_private_texts()

    # Verify collection was created (this creates a fresh singleton)
    qdrant = get_qdrant_client()
    if not qdrant.collection_exists(PRIVATE_COLLECTION):
        raise RuntimeError(f"Failed to create collection {PRIVATE_COLLECTION}")
    count = qdrant.get_collection(PRIVATE_COLLECTION).points_count or 0
    print(f"  Private collection created ({count} points)")


def _run_eval(
    limit: int | None,
    verbose: bool,
    parallel: bool,
    workers: int,
    no_judge: bool,
    output: str | None,
    debug: bool,
    eval_mode: str,
) -> None:
    """Run the flow evaluation."""
    eval_start_time = time.time()

    # Check if Qdrant is accessible
    if not check_qdrant_access():
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
        return

    # Warmup: trigger model loading (suppress expected "no results" warning)
    console.print("\n[dim]Warming up models...[/dim]")
    try:
        from backend.agent.rag.tools import tool_account_rag

        # Temporarily suppress RAG warnings during warmup (expected to fail with fake company)
        rag_logger = logging.getLogger("backend.agent.rag.tools")
        original_level = rag_logger.level
        rag_logger.setLevel(logging.ERROR)
        try:
            tool_account_rag("warmup", "test_company", top_k=1)
        finally:
            rag_logger.setLevel(original_level)
        console.print("[dim]Models loaded.[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Model preload failed: {e}[/yellow]")

    # Show tree stats
    stats = get_tree_stats()
    console.print("\n[bold]Question Tree Stats:[/bold]")
    for key, value in stats.items():
        console.print(f"  [dim]{key}:[/dim] {value}")

    # Run evaluation (synchronous - uses ThreadPoolExecutor internally)
    use_judge = not no_judge
    concurrency = workers if parallel else 1
    try:
        results = run_flow_eval(
            max_paths=limit,
            verbose=verbose,
            use_judge=use_judge,
            concurrency=concurrency,
            eval_mode=eval_mode,
        )
    except Exception as e:
        console.print(f"\n[red bold]ERROR: Evaluation failed: {e}[/red bold]")
        import traceback

        traceback.print_exc()
        return

    # Fetch latency breakdown from LangSmith
    elapsed_minutes = int((time.time() - eval_start_time) / 60) + 1
    latency_pcts = get_latency_percentages(minutes_ago=max(elapsed_minutes, 5))
    if latency_pcts:
        results.latency_routing_pct = latency_pcts.get("routing", 0.0)
        results.latency_retrieval_pct = latency_pcts.get("retrieval", 0.0)
        results.latency_answer_pct = latency_pcts.get("answer", 0.0)

    # Print summary
    print_summary(results, eval_mode=eval_mode)

    # Debug output for failing paths
    if debug and results.failed_paths:

        def format_failed_path(i: int, item: dict) -> None:
            console.print(f"\n[bold cyan]--- Path {item['path_id']} ---[/bold cyan]")
            for j, step in enumerate(item["steps"]):
                status = "PASS" if step["passed"] else "FAIL"
                console.print(f"[bold]Q{j + 1}:[/bold] {step['question']}")
                console.print(f"    [{status}] R={step['relevance_score']} F={step['faithfulness_score']}")
                console.print(f"    [bold]Answer:[/bold] {step['answer'][:200]}...")
                if step["judge_explanation"]:
                    console.print(f"    [bold]Judge:[/bold] {step['judge_explanation']}")

        print_debug_failures(
            [
                {
                    "path_id": fp.path_id,
                    "steps": [
                        {
                            "question": s.question,
                            "passed": s.passed,
                            "relevance_score": s.relevance_score,
                            "faithfulness_score": s.faithfulness_score,
                            "answer": s.answer,
                            "judge_explanation": s.judge_explanation,
                        }
                        for s in fp.steps
                    ],
                }
                for fp in results.failed_paths
            ],
            title="Full details for failed paths",
            format_item=format_failed_path,
        )

    # Save if requested
    if output:
        save_results(results, Path(output))


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
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing paths"),
    eval_mode: str = typer.Option(
        "both",
        "--eval-mode",
        "-e",
        help="RAGAS mode: 'rag' (precision/recall), 'pipeline' (faithfulness/relevance), or 'both'",
    ),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    _run_eval(
        limit=limit,
        verbose=verbose,
        parallel=parallel,
        workers=workers,
        no_judge=no_judge,
        output=output,
        debug=debug,
        eval_mode=eval_mode,
    )


if __name__ == "__main__":
    print("Checking Qdrant collections...")
    ensure_qdrant_collections()
    print()
    app()
