"""
CLI for question tree utilities.

Usage:
    python -m backend.agent.question_tree validate [--role sales|csm|manager]
    python -m backend.agent.question_tree tree [--role sales|csm|manager] [--depth N]
    python -m backend.agent.question_tree stats [--role sales|csm|manager]
    python -m backend.agent.question_tree paths --role sales
"""

from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import get_paths_for_role, get_tree_stats, print_tree, validate_tree

app = typer.Typer(help="Question tree utilities for demo reliability.")
console = Console()

ROLE_LABELS = {
    "sales": "SALES REP (jsmith)",
    "csm": "CSM (amartin)",
    "manager": "MANAGER",
}

T = TypeVar("T")


def _call_with_role(func: Callable[..., T], role: str | None, **kwargs: object) -> T:
    """Call a function with role, handling ValueError consistently."""
    try:
        return func(role=role, **kwargs)
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def validate(
    role: str | None = typer.Option(None, "--role", "-r", help="Filter by role: sales, csm, or manager"),
) -> None:
    """Validate the question tree for consistency."""
    role_label = role.upper() if role else "ALL"
    issues = _call_with_role(validate_tree, role)

    if issues:
        rprint(f"[red]Validation failed for {role_label}![/red]")
        for issue in issues:
            rprint(f"  [red]x[/red] {issue}")
        raise typer.Exit(1)
    else:
        rprint(f"[green]OK[/green] {role_label} tree is valid")


@app.command()
def tree(
    role: str | None = typer.Option(None, "--role", "-r", help="Filter by role: sales, csm, or manager"),
    depth: int | None = typer.Option(None, "--depth", "-d", help="Max depth to display"),
) -> None:
    """Print the question tree in a top-down format."""
    tree_output = print_tree(role=role, max_depth=depth)
    console.print(tree_output)


@app.command()
def stats(
    role: str | None = typer.Option(None, "--role", "-r", help="Filter by role: sales, csm, or manager"),
) -> None:
    """Show statistics about the question tree."""
    s = _call_with_role(get_tree_stats, role)
    role_label = s["role"].upper()
    table = Table(title=f"Question Tree Statistics ({role_label})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Starter questions", str(s["num_starters"]))
    table.add_row("Total questions", str(s["num_questions"]))
    table.add_row("Edges (follow-up links)", str(s["num_edges"]))
    table.add_row("Max depth", str(s["max_depth"]))
    table.add_row("Total paths", str(s["num_paths"]))
    table.add_row("Min path length", str(s["path_lengths"]["min"]))
    table.add_row("Max path length", str(s["path_lengths"]["max"]))

    console.print(table)


@app.command()
def paths(
    role: str | None = typer.Option(None, "--role", "-r", help="Filter by role: sales, csm, or manager"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of paths shown"),
) -> None:
    """List all conversation paths for auditing workflows."""
    all_paths = _call_with_role(get_paths_for_role, role)
    role_label = ROLE_LABELS.get(role, "ALL ROLES") if role else "ALL ROLES"

    # Group paths by depth
    paths_by_depth: dict[int, list] = defaultdict(list)
    for path in all_paths:
        paths_by_depth[len(path)].append(path)

    console.print(f"\n[bold]{role_label}[/bold] - {len(all_paths)} paths\n")

    path_num = 0
    for depth in sorted(paths_by_depth.keys()):
        console.print(f"[dim]-- Depth {depth} ({len(paths_by_depth[depth])} paths) --[/dim]\n")

        for path in paths_by_depth[depth]:
            path_num += 1
            if limit and path_num > limit:
                remaining = len(all_paths) - limit
                console.print(f"\n[dim]... and {remaining} more paths (use --limit to see more)[/dim]")
                return

            # Format path as numbered steps
            steps = []
            for i, question in enumerate(path, 1):
                steps.append(f"[cyan]{i}.[/cyan] {question}")

            path_content = "\n".join(steps)
            console.print(Panel(path_content, title=f"Path {path_num}", border_style="dim"))
            console.print()


if __name__ == "__main__":
    app()
