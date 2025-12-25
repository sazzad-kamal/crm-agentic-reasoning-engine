"""
Base evaluation utilities shared between docs and account eval.

Provides:
- Rich console formatting helpers
- Common summary table rendering
- Shared metrics computation
"""

from typing import Callable
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


# Shared console instance
console = Console()


def create_summary_table(title: str = "Evaluation Summary") -> Table:
    """Create a standard summary table with consistent styling."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    return table


def create_detail_table(title: str, columns: list[tuple[str, str]]) -> Table:
    """
    Create a detail table with custom columns.
    
    Args:
        title: Table title
        columns: List of (name, justify) tuples
    """
    table = Table(title=title, show_header=True, header_style="bold")
    for name, justify in columns:
        table.add_column(name, justify=justify)
    return table


def print_eval_header(title: str, subtitle: str) -> None:
    """Print evaluation header panel."""
    console.print(Panel(
        subtitle,
        title=title,
        border_style="blue",
    ))


def print_issues_panel(title: str, issues: list[str]) -> None:
    """Print issues panel if there are any issues."""
    if issues:
        console.print(Panel(
            "\n\n".join(issues),
            title=f"[red]{title}[/red]",
            border_style="red",
        ))


def format_check_mark(value: bool) -> str:
    """Format boolean as colored check/cross mark."""
    return "[green]✓[/green]" if value else "[red]✗[/red]"


def add_separator_row(table: Table) -> None:
    """Add a visual separator row to a table."""
    table.add_row("─" * 20, "─" * 10)
