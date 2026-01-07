"""
Formatting utilities for eval output.

Rich console formatting helpers for tables, panels, and styled output.
"""

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Shared console instance
console = Console()


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


def create_summary_table(title: str = "Evaluation Summary") -> Table:
    """Create a standard summary table with consistent styling."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    return table


def print_eval_header(title: str, subtitle: str) -> None:
    """Print evaluation header panel."""
    console.print(
        Panel(
            subtitle,
            title=title,
            border_style="blue",
        )
    )


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
    failures: list[dict[str, Any]],
    title: str,
    max_items: int = 10,
    format_item: Callable[[int, dict[str, Any]], None] | None = None,
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

    for section_name, rows in sections:
        if section_name:
            # Color section header based on whether all SLOs in section pass
            section_passed = all(slo_passed for _, _, _, slo_passed in rows if slo_passed is not None)
            color = "green" if section_passed else "red"
            table.add_row(f"[bold {color}]{section_name}[/bold {color}]", "", "")

        for label, value, slo_target, _slo_passed in rows:
            slo_display = slo_target if slo_target else ""
            table.add_row(label, value, slo_display)

        table.add_row("", "", "")  # Spacer

    return table


__all__ = [
    "console",
    "format_check_mark",
    "format_percentage",
    "create_summary_table",
    "print_eval_header",
    "print_overall_result_panel",
    "print_debug_failures",
    "build_eval_table",
]
