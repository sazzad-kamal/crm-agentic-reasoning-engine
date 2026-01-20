"""CLI for integration evaluation."""

from __future__ import annotations

import asyncio
import platform
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

# Fix Windows asyncio cleanup issues with httpx/RAGAS
# Use SelectorEventLoop instead of ProactorEventLoop to avoid "Event loop is closed" errors
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

import atexit
import logging
import time

import typer
from rich.panel import Panel

from backend.agent.fetch.rag.client import close_qdrant_client
from backend.eval.integration.langsmith import get_latency_percentages
from backend.eval.integration.output import check_qdrant_access, print_summary, save_results
from backend.eval.integration.runner import run_flow_eval
from backend.eval.integration.tree import get_tree_stats
from backend.eval.shared.formatting import console

# Register cleanup to prevent shutdown errors
atexit.register(close_qdrant_client)

app = typer.Typer()


def ensure_qdrant_collections() -> None:
    """Ensure Qdrant collections exist, ingesting data if needed."""
    from backend.agent.fetch.rag.client import close_qdrant_client, get_qdrant_client
    from backend.agent.fetch.rag.config import QDRANT_PATH, TEXT_COLLECTION
    from backend.agent.fetch.rag.ingest import ingest_texts

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant = get_qdrant_client()

    collection_exists = (
        qdrant.collection_exists(TEXT_COLLECTION)
        and (qdrant.get_collection(TEXT_COLLECTION).points_count or 0) > 0
    )

    if collection_exists:
        print("Qdrant collections ready.")
        return

    # Close singleton before ingest (ingest needs exclusive access to local storage)
    close_qdrant_client()

    print("Ingesting texts into Qdrant...")
    ingest_texts()

    # Verify collection was created (this creates a fresh singleton)
    qdrant = get_qdrant_client()
    if not qdrant.collection_exists(TEXT_COLLECTION):
        raise RuntimeError(f"Failed to create collection {TEXT_COLLECTION}")
    count = qdrant.get_collection(TEXT_COLLECTION).points_count or 0
    print(f"  Text collection created ({count} points)")


def _run_eval(
    limit: int | None,
    verbose: bool,
    no_judge: bool,
    output: str | None,
    debug: bool,
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
                "  2. Close any Jupyter notebooks using RAG",
                border_style="red",
            )
        )
        return

    # Warmup: trigger model loading (suppress expected "no results" warning)
    console.print("\n[dim]Warming up models...[/dim]")
    try:
        from backend.agent.fetch.rag.search import search_entity_context

        # Temporarily suppress RAG warnings during warmup (expected to fail with fake company)
        rag_logger = logging.getLogger("backend.agent.fetch.rag.search")
        original_level = rag_logger.level
        rag_logger.setLevel(logging.ERROR)
        try:
            search_entity_context("warmup", {"company_id": "test_company"})
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

    # Run evaluation (sequential)
    use_judge = not no_judge
    try:
        results = run_flow_eval(
            max_paths=limit,
            verbose=verbose,
            use_judge=use_judge,
            concurrency=1,
        )
    except Exception as e:
        console.print(f"\n[red bold]ERROR: Evaluation failed: {e}[/red bold]")
        import traceback

        traceback.print_exc()
        return

    # Fetch latency breakdown from LangSmith (optional, informational only)
    elapsed_minutes = int((time.time() - eval_start_time) / 60) + 1
    latency_pcts = get_latency_percentages(minutes_ago=max(elapsed_minutes, 5))

    # Print summary with optional LangSmith info
    print_summary(results, latency_pcts=latency_pcts)

    # Debug output for failing paths
    if debug and results.failed_paths:
        console.print("\n[bold yellow]DEBUG: Failed paths[/bold yellow]")
        for fp in results.failed_paths[:10]:
            console.print(f"\n[bold cyan]--- Path {fp.path_id} ---[/bold cyan]")
            for j, s in enumerate(fp.steps):
                status = "PASS" if s.passed else "FAIL"
                console.print(f"[bold]Q{j + 1}:[/bold] {s.question}")
                console.print(f"    [{status}] R={s.relevance_score} F={s.faithfulness_score}")
                console.print(f"    [bold]Answer:[/bold] {s.answer[:200]}...")
                if s.judge_explanation:
                    console.print(f"    [bold]Judge:[/bold] {s.judge_explanation}")

    # Save if requested
    if output:
        save_results(results, Path(output))


@app.command()
def main(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths to test"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output for each question"
    ),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge evaluation"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save JSON results"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Dump full details for failing paths"),
) -> None:
    """Run conversation flow evaluation."""
    logging.basicConfig(level=logging.WARNING)
    _run_eval(
        limit=limit,
        verbose=verbose,
        no_judge=no_judge,
        output=output,
        debug=debug,
    )


if __name__ == "__main__":
    print("Checking Qdrant collections...")
    ensure_qdrant_collections()
    print()
    app()
