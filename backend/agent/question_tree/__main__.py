"""
CLI for question tree utilities.

Usage:
    python -m backend.agent.question_tree validate
    python -m backend.agent.question_tree mermaid
    python -m backend.agent.question_tree stats
"""

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from . import get_tree_stats, to_mermaid, validate_tree

app = typer.Typer(help="Question tree utilities for demo reliability.")
console = Console()


@app.command()
def validate() -> None:
    """Validate the question tree for consistency."""
    issues = validate_tree()
    if issues:
        rprint("[red]Validation failed![/red]")
        for issue in issues:
            rprint(f"  [red]x[/red] {issue}")
        raise typer.Exit(1)
    else:
        rprint("[green]OK[/green] Tree is valid")


@app.command()
def mermaid(max_label: int = 40) -> None:
    """Generate a Mermaid diagram of the tree."""
    diagram = to_mermaid(max_label_length=max_label)
    print(diagram)


@app.command()
def stats() -> None:
    """Show statistics about the question tree."""
    s = get_tree_stats()

    table = Table(title="Question Tree Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Starter questions", str(s["num_starters"]))
    table.add_row("Total questions", str(s["num_questions"]))
    table.add_row("Edges (follow-up links)", str(s["num_edges"]))
    table.add_row("Total paths", str(s["num_paths"]))
    table.add_row("Min path length", str(s["path_lengths"]["min"]))
    table.add_row("Max path length", str(s["path_lengths"]["max"]))

    console.print(table)


if __name__ == "__main__":
    app()
