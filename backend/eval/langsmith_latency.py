"""LangSmith latency breakdown for eval runs."""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta

from rich.table import Table

from backend.eval.formatting import console


def get_latency_breakdown(
    minutes_ago: int = 10,
    project_name: str | None = None,
    limit: int = 500,
) -> dict[str, dict]:
    """
    Fetch latency breakdown by node from LangSmith.

    Args:
        minutes_ago: Look back this many minutes for runs
        project_name: LangSmith project name (defaults to LANGCHAIN_PROJECT env var)

    Returns:
        Dict mapping node_name -> {count, total_ms, avg_ms, min_ms, max_ms}
    """
    try:
        from langsmith import Client
    except ImportError:
        console.print("[yellow]langsmith not installed. Run: pip install langsmith[/yellow]")
        return {}

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        console.print("[yellow]LANGCHAIN_API_KEY not set - skipping latency breakdown[/yellow]")
        return {}

    project = project_name or os.getenv("LANGCHAIN_PROJECT", "default")
    client = Client()

    # Get runs from recent time window
    start_time = datetime.utcnow() - timedelta(minutes=minutes_ago)

    try:
        # Fetch runs in a single API call with limit
        runs = list(client.list_runs(
            project_name=project,
            start_time=start_time,
            limit=limit,
        ))
    except Exception as e:
        console.print(f"[yellow]Could not fetch LangSmith runs: {e}[/yellow]")
        return {}

    if not runs:
        console.print(f"[dim]No runs found in last {minutes_ago} minutes[/dim]")
        return {}

    # Aggregate latencies by node name (filter to child runs only)
    node_latencies: dict[str, list[float]] = defaultdict(list)

    for run in runs:
        # Only include runs that have a parent (child nodes)
        if run.parent_run_id and run.end_time and run.start_time:
            latency_ms = (run.end_time - run.start_time).total_seconds() * 1000
            node_latencies[run.name].append(latency_ms)

    # Compute stats
    breakdown = {}
    for node, latencies in node_latencies.items():
        if latencies:
            breakdown[node] = {
                "count": len(latencies),
                "total_ms": sum(latencies),
                "avg_ms": sum(latencies) / len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
            }

    return breakdown


AGENT_NODES = {"route", "fetch_crm", "fetch_docs", "fetch_account", "answer", "followup"}


def print_latency_breakdown(
    minutes_ago: int = 10,
    project_name: str | None = None,
) -> None:
    """Print latency breakdown table from LangSmith data."""
    breakdown = get_latency_breakdown(minutes_ago, project_name, limit=500)

    if not breakdown:
        return

    # Filter to only agent nodes (exclude RAGAS evaluation nodes like "row 0")
    breakdown = {k: v for k, v in breakdown.items() if k in AGENT_NODES}

    if not breakdown:
        console.print("[dim]No agent node latencies found[/dim]")
        return

    # Calculate total for percentages
    total_avg = sum(stats["avg_ms"] for stats in breakdown.values())

    table = Table(title="Node Latency Breakdown (LangSmith)", show_header=True)
    table.add_column("Node", style="bold")
    table.add_column("Calls", justify="right")
    table.add_column("Avg", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("%", justify="right")

    # Sort by average latency descending
    for node, stats in sorted(breakdown.items(), key=lambda x: -x[1]["avg_ms"]):
        pct = (stats["avg_ms"] / total_avg * 100) if total_avg > 0 else 0
        table.add_row(
            node,
            str(stats["count"]),
            f"{stats['avg_ms']:.0f}ms",
            f"{stats['min_ms']:.0f}ms",
            f"{stats['max_ms']:.0f}ms",
            f"{pct:.1f}%",
        )

    console.print()
    console.print(table)
    console.print(f"[dim]Total avg: {total_avg:.0f}ms across {len(breakdown)} nodes[/dim]")
