"""
Shared evaluation utilities for agent evaluation harnesses.

Provides:
- Rich console formatting helpers
- Summary table rendering
- Baseline comparison utilities
- Parallel evaluation runner
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar, Callable, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

T = TypeVar("T")

# Shared console instance
console = Console()


# =============================================================================
# Table Creation
# =============================================================================


def create_summary_table(title: str = "Evaluation Summary") -> Table:
    """Create a standard summary table with consistent styling."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    return table


# =============================================================================
# Formatting Helpers
# =============================================================================


def format_check_mark(value: bool) -> str:
    """Format boolean as colored check/cross mark."""
    return "[green]Y[/green]" if value else "[red]X[/red]"


def format_percentage(value: float, thresholds: tuple[float, float] = (0.9, 0.7)) -> str:
    """
    Format percentage with color based on thresholds.

    Args:
        value: Float between 0 and 1
        thresholds: Tuple of (green_threshold, yellow_threshold)

    Returns:
        Colored percentage string
    """
    green_thresh, yellow_thresh = thresholds
    if value >= green_thresh:
        color = "green"
    elif value >= yellow_thresh:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{value:.1%}[/{color}]"


# =============================================================================
# Panel Helpers
# =============================================================================


def print_eval_header(title: str, subtitle: str) -> None:
    """Print evaluation header panel."""
    console.print(
        Panel(
            subtitle,
            title=title,
            border_style="blue",
        )
    )


# =============================================================================
# Baseline Comparison
# =============================================================================

REGRESSION_THRESHOLD = 0.05  # 5% regression threshold


def compare_to_baseline(
    current_score: float,
    baseline_path: Path,
    score_key: str = "overall_score",
) -> tuple[bool, float | None]:
    """
    Compare current score to a baseline.

    Args:
        current_score: Current evaluation score
        baseline_path: Path to baseline JSON file
        score_key: Key in baseline JSON containing the score

    Returns:
        Tuple of (is_regression, baseline_score)
    """
    if not baseline_path.exists():
        return False, None

    try:
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        # Handle nested summary structure
        if "summary" in baseline_data:
            baseline_score = baseline_data["summary"].get(score_key, 0.0)
        else:
            baseline_score = baseline_data.get(score_key, 0.0)

        # Regression if we're more than threshold worse
        is_regression = current_score < (baseline_score - REGRESSION_THRESHOLD)

        return is_regression, baseline_score
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load baseline: {e}[/yellow]")
        return False, None


def save_baseline(
    summary: dict[str, Any],
    baseline_path: Path,
) -> None:
    """
    Save current results as new baseline.

    Args:
        summary: Summary dictionary to save
        baseline_path: Path to save baseline JSON
    """
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump({"summary": summary}, f, indent=2, default=str)
    console.print(f"[green]Baseline saved to {baseline_path}[/green]")


def print_baseline_comparison(
    current_score: float,
    baseline_score: float | None,
    is_regression: bool,
) -> None:
    """
    Print baseline comparison results.

    Args:
        current_score: Current evaluation score
        baseline_score: Baseline score (or None if no baseline)
        is_regression: Whether regression was detected
    """
    if baseline_score is None:
        console.print("[dim]No baseline found for comparison[/dim]")
        return

    diff = current_score - baseline_score
    diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"

    if is_regression:
        console.print("\n[red bold]REGRESSION DETECTED[/red bold]")
        console.print(
            f"  Baseline: {baseline_score:.1%} -> Current: {current_score:.1%} ({diff_str})"
        )
    else:
        color = "green" if diff >= 0 else "yellow"
        console.print(
            f"\n[dim]Baseline: {baseline_score:.1%} -> Current: {current_score:.1%} ([{color}]{diff_str}[/{color}])[/dim]"
        )


# =============================================================================
# Parallel Evaluation Runner
# =============================================================================


def run_parallel_evaluation(
    items: list[dict],
    evaluate_fn: Callable[[dict, Optional[threading.Lock]], T],
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
            future_to_item = {executor.submit(evaluate_with_lock, item): item for item in items}

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


# =============================================================================
# LLM Judge Utilities
# =============================================================================


def parse_json_response(text: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as dictionary

    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


def run_llm_judge(
    prompt: str,
    system_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict:
    """
    Run LLM judge and parse JSON response.

    Args:
        prompt: The evaluation prompt
        system_prompt: System prompt for the judge
        model: Model to use
        temperature: Temperature for generation
        max_tokens: Max tokens for response

    Returns:
        Parsed JSON result, or dict with 'error' key on failure
    """
    from backend.agent.eval.llm_client import call_llm

    try:
        response = call_llm(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response or not response.strip():
            return {"error": "Empty response from judge LLM"}

        return parse_json_response(response)
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# SLO Table Creation
# =============================================================================


def create_slo_table(
    slo_checks: list[tuple[str, bool, str, str]],
    title: str = "SLO Summary",
) -> Table:
    """
    Create a standard SLO results table.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples
        title: Table title

    Returns:
        Formatted Rich Table
    """
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("SLO", style="bold")
    table.add_column("Actual", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_column("Status", justify="center")

    for name, passed, actual, target in slo_checks:
        table.add_row(name, actual, target, format_check_mark(passed))

    return table


def print_slo_result(slo_checks: list[tuple[str, bool, str, str]]) -> bool:
    """
    Print SLO summary table and pass/fail status.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples

    Returns:
        True if all SLOs passed, False otherwise
    """
    console.print()
    slo_table = create_slo_table(slo_checks)
    console.print(slo_table)

    passed_slos = sum(1 for _, passed, _, _ in slo_checks if passed)
    total_slos = len(slo_checks)
    failed_slo_names = [name for name, passed, _, _ in slo_checks if not passed]

    if failed_slo_names:
        console.print(f"\n[red bold][!] {len(failed_slo_names)} SLO(s) FAILED:[/red bold]")
        for slo_name in failed_slo_names:
            console.print(f"    [red]X[/red] {slo_name}")
        return False
    else:
        console.print(f"\n[green bold][OK] All {total_slos} SLOs passed[/green bold]")
        return True


# =============================================================================
# Latency Calculations
# =============================================================================


def calculate_p95_latency(latencies: list[float | int]) -> float:
    """
    Calculate P95 latency from a list of latencies.

    Args:
        latencies: List of latency values in milliseconds

    Returns:
        P95 latency value, or 0.0 if list is empty
    """
    if not latencies:
        return 0.0

    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)
    return float(sorted_latencies[min(p95_index, len(sorted_latencies) - 1)])


# =============================================================================
# Exit Code Logic
# =============================================================================


def determine_exit_code(
    all_slos_passed: bool,
    is_regression: bool,
) -> int:
    """
    Determine exit code from SLO and regression status.

    Args:
        all_slos_passed: Whether all SLOs passed
        is_regression: Whether a regression was detected

    Returns:
        0 for success, 1 for failure
    """
    if not all_slos_passed or is_regression:
        return 1
    return 0


# =============================================================================
# Result Panel Printing
# =============================================================================


def get_failed_slos(slo_checks: list[tuple[str, bool, str, str]]) -> list[str]:
    """
    Extract names of failed SLOs from slo_checks list.

    Args:
        slo_checks: List of (name, passed, actual_value, target_value) tuples

    Returns:
        List of SLO names that failed
    """
    return [name for name, passed, _, _ in slo_checks if not passed]


def print_overall_result_panel(
    all_passed: bool,
    failure_reasons: list[str],
    success_message: str,
) -> None:
    """
    Print overall pass/fail result panel.

    Args:
        all_passed: Whether overall evaluation passed
        failure_reasons: List of failure reason strings
        success_message: Message to show on success
    """
    if all_passed:
        console.print(
            Panel(
                f"[green bold]OVERALL: PASS[/green bold]\n{success_message}",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[red bold]OVERALL: FAIL[/red bold]\n{'; '.join(failure_reasons)}",
                border_style="red",
            )
        )


def print_debug_failures(
    failures: list[dict],
    title: str,
    max_items: int = 10,
    format_item: Callable[[int, dict], None] | None = None,
) -> None:
    """
    Print debug output for failed items.

    Args:
        failures: List of failure dictionaries
        title: Title for the debug section
        max_items: Maximum number of items to show
        format_item: Optional function to format each item (receives index, item)
    """
    if not failures:
        return

    console.print("\n" + "=" * 80)
    console.print(f"[bold yellow]DEBUG: {title}[/bold yellow]")
    console.print("=" * 80)

    for i, item in enumerate(failures[:max_items]):
        if format_item:
            format_item(i, item)
        else:
            console.print(f"\n[bold cyan]--- Item {i + 1} ---[/bold cyan]")
            for key, value in item.items():
                console.print(f"[bold]{key}:[/bold] {value}")
        console.print("-" * 40)


# =============================================================================
# Generic Eval CLI Finalization
# =============================================================================


def finalize_eval_cli(
    primary_score: float,
    slo_checks: list[tuple[str, bool, str, str]],
    baseline_path: Path,
    score_key: str,
    set_baseline: bool = False,
    baseline_data: dict | None = None,
    extra_failure_check: bool = False,
    extra_failure_reason: str = "",
) -> int:
    """
    Finalize evaluation CLI: handle baseline, SLOs, and exit code.

    This is a shared helper for eval CLIs to avoid duplication.

    Args:
        primary_score: The primary score to compare against baseline
        slo_checks: List of (name, passed, actual_value, target_value) tuples
        baseline_path: Path to baseline JSON file
        score_key: Key for score in baseline JSON
        set_baseline: Whether to save current results as baseline
        baseline_data: Data to save as baseline (required if set_baseline=True)
        extra_failure_check: Additional failure condition (e.g., paths_failed > 0)
        extra_failure_reason: Reason for extra failure

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Baseline comparison
    is_regression, baseline_score = compare_to_baseline(
        primary_score,
        baseline_path,
        score_key=score_key,
    )
    print_baseline_comparison(primary_score, baseline_score, is_regression)

    # Save as new baseline if requested
    if set_baseline and baseline_data:
        save_baseline(baseline_data, baseline_path)

    # Check SLOs
    failed_slos = get_failed_slos(slo_checks)
    all_slos_passed = len(failed_slos) == 0
    exit_code = determine_exit_code(all_slos_passed, is_regression)

    # Include extra failure condition
    if extra_failure_check:
        exit_code = 1

    # Build failure reasons
    failure_reasons = []
    if extra_failure_check and extra_failure_reason:
        failure_reasons.append(extra_failure_reason)
    if failed_slos:
        failure_reasons.append(f"{len(failed_slos)} SLOs failed: {', '.join(failed_slos)}")
    if is_regression:
        failure_reasons.append("Regression detected vs baseline")

    # Print overall result panel
    console.print()
    print_overall_result_panel(
        all_passed=exit_code == 0,
        failure_reasons=failure_reasons,
        success_message=f"All {len(slo_checks)} SLOs met, no regression detected",
    )

    return exit_code


def save_eval_results(
    output_path: str,
    summary: Any,
    results: list[Any],
    result_mapper: Callable[[Any], dict],
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        output_path: Path to save JSON results
        summary: Summary object (must have model_dump() or be dict)
        results: List of result objects
        result_mapper: Function to convert result object to dict
    """
    summary_dict = summary.model_dump() if hasattr(summary, "model_dump") else summary
    output_data = {
        "summary": summary_dict,
        "results": [result_mapper(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    console.print(f"[dim]Results saved to {output_path}[/dim]")


# =============================================================================
# Generic Eval Summary Table Builder
# =============================================================================


def build_eval_table(
    title: str,
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]],
) -> Table:
    """
    Build a standardized evaluation summary table.

    Args:
        title: Table title
        sections: List of (section_name, rows) where rows are
                  (label, value, slo_target, slo_passed)
                  slo_target=None means "tracked", slo_passed=None means no status

    Returns:
        Rich Table
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("SLO", justify="right", style="dim")
    table.add_column("Status", justify="center")

    for section_name, rows in sections:
        if section_name:
            table.add_row(f"[bold]{section_name}[/bold]", "", "", "")

        for label, value, slo_target, slo_passed in rows:
            slo_display = slo_target if slo_target else "[dim]tracked[/dim]"
            status = format_check_mark(slo_passed) if slo_passed is not None else ""
            table.add_row(label, value, slo_display, status)

        table.add_row("", "", "", "")  # Spacer

    return table


__all__ = [
    "console",
    "create_summary_table",
    "format_check_mark",
    "format_percentage",
    "print_eval_header",
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
    "run_parallel_evaluation",
    "REGRESSION_THRESHOLD",
    # LLM Judge utilities
    "parse_json_response",
    "run_llm_judge",
    # SLO utilities
    "create_slo_table",
    "print_slo_result",
    "get_failed_slos",
    # Latency utilities
    "calculate_p95_latency",
    # Exit and result utilities
    "determine_exit_code",
    "print_overall_result_panel",
    "print_debug_failures",
    # Generic eval CLI utilities
    "finalize_eval_cli",
    "save_eval_results",
    "build_eval_table",
]
