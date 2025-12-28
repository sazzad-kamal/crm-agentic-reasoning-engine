"""
Shared parallel execution utilities for RAG evaluation.

Provides a reusable ThreadPoolExecutor-based runner with progress tracking.
Used by docs_eval.py and account_eval.py to eliminate code duplication.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar, Callable, Any

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from backend.rag.eval.base import console


T = TypeVar("T")


def run_parallel_evaluation(
    items: list[dict],
    evaluate_fn: Callable[[dict, threading.Lock | None], T],
    max_workers: int,
    description: str,
    id_field: str = "id",
    use_lock: bool = True,
) -> list[T]:
    """
    Run evaluation in parallel using ThreadPoolExecutor.

    Provides thread-safe access via an optional lock for non-thread-safe
    components (like embedding models).

    Args:
        items: List of items to evaluate (each must have an id field)
        evaluate_fn: Function that takes (item, lock) and returns result
        max_workers: Maximum number of parallel workers
        description: Description for progress bar
        id_field: Name of the field containing item ID (default: "id")
        use_lock: Whether to use a lock for thread-safe access (default: True)

    Returns:
        List of results in the same order as input items
    """
    total = len(items)
    results_by_id: dict[str, T] = {}

    # Lock for thread-safe access
    lock = threading.Lock() if use_lock else None

    def evaluate_with_lock(item: dict) -> T:
        """Wrapper that passes lock to evaluate function."""
        return evaluate_fn(item, lock)

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
            f"{description} ({total} items, max {max_workers} workers)",
            total=total,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(evaluate_with_lock, item): item
                for item in items
            }

            # Process as they complete
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                item_id = item[id_field]
                try:
                    result = future.result()
                    results_by_id[item_id] = result
                except Exception as e:
                    progress.console.print(f"  [red]✗ {item_id}: {e}[/red]")
                finally:
                    progress.advance(task)

    # Return in original order
    results = []
    for item in items:
        item_id = item[id_field]
        if item_id in results_by_id:
            results.append(results_by_id[item_id])

    return results


__all__ = ["run_parallel_evaluation"]
