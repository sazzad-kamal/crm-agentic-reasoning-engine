"""
Formatting utilities for eval output.

Rich console formatting helpers for tables and styled output.
"""

from rich.console import Console
from rich.table import Table

# Shared console instance
console = Console()


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


def build_eval_table(
    title: str,
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]],
    aggregate_row: tuple[str, str, str, bool] | None = None,
) -> Table:
    """
    Build a standardized evaluation summary table.

    Args:
        title: Table title
        sections: List of (section_name, rows) where rows are
                  (label, value, slo_target, slo_passed)
                  slo_target=None means "tracked", slo_passed=None means no status
        aggregate_row: Optional (label, value, slo_target, slo_passed) for bottom row with border

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

    # Add aggregate row with separator if provided
    if aggregate_row:
        label, value, slo_target, slo_passed = aggregate_row
        color = "green" if slo_passed else "red"
        table.add_row("─" * 20, "─" * 10, "─" * 10, end_section=True)
        table.add_row(
            f"[bold {color}]{label}[/bold {color}]",
            f"[bold {color}]{value}[/bold {color}]",
            slo_target,
        )

    return table


__all__ = [
    "console",
    "format_percentage",
    "build_eval_table",
]
