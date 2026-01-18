"""LangSmith latency breakdown for eval runs."""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta

from backend.eval.shared.formatting import console


def get_latency_breakdown(
    minutes_ago: int = 10,
    project_name: str | None = None,
    limit: int = 1000,
) -> dict[str, dict]:
    """
    Fetch latency breakdown by node from LangSmith.

    Args:
        minutes_ago: Look back this many minutes for runs
        project_name: LangSmith project name (defaults to LANGCHAIN_PROJECT env var)
        limit: Max runs to fetch (higher = more complete but slower)

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
        # Fetch only child runs (is_root=False) to exclude RAGAS top-level runs
        runs = list(client.list_runs(
            project_name=project,
            start_time=start_time,
            is_root=False,
            limit=limit,
        ))
    except Exception as e:
        console.print(f"[yellow]Could not fetch LangSmith runs: {e}[/yellow]")
        return {}

    if not runs:
        console.print(f"[dim]No runs found in last {minutes_ago} minutes[/dim]")
        return {}

    # Aggregate latencies by node name
    node_latencies: dict[str, list[float]] = defaultdict(list)

    for run in runs:
        if run.end_time and run.start_time:
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


AGENT_NODES = {"fetch", "answer", "followup"}


def get_latency_percentages(
    minutes_ago: int = 10,
    project_name: str | None = None,
) -> dict[str, float]:
    """
    Get latency percentages by section for display in eval summary.

    Returns dict with keys: routing, retrieval, answer, followup
    Values are percentages (0.0 to 1.0)
    """
    breakdown = get_latency_breakdown(minutes_ago, project_name)

    if not breakdown:
        return {}

    # Filter to agent nodes only
    breakdown = {k: v for k, v in breakdown.items() if k in AGENT_NODES}

    if not breakdown:
        return {}

    # Calculate total for percentages
    total_avg = sum(stats["avg_ms"] for stats in breakdown.values())

    if total_avg == 0:
        return {}

    # Map node names to sections
    fetch_pct = breakdown.get("fetch", {}).get("avg_ms", 0) / total_avg
    answer_pct = breakdown.get("answer", {}).get("avg_ms", 0) / total_avg
    followup_pct = breakdown.get("followup", {}).get("avg_ms", 0) / total_avg

    return {
        "fetch": fetch_pct,
        "answer": answer_pct,
        "followup": followup_pct,
    }
