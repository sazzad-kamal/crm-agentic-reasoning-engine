"""
Agent Evaluation History Tracking.

Stores multiple agent evaluation runs and provides trend analysis.
Mirrors backend.rag.eval.history for consistency.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.panel import Panel

from backend.agent.eval.models import (
    E2EEvalSummary,
    SLO_EVAL_LATENCY_P95_MS,
    SLO_EVAL_LATENCY_AVG_MS,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
    SLO_MODE_ACCURACY,
)
from backend.agent.eval.base import console


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
HISTORY_FILE = DATA_DIR / "agent_eval_history.json"
MAX_HISTORY_ENTRIES = 20


# =============================================================================
# History Storage
# =============================================================================

def load_agent_history() -> list[dict[str, Any]]:
    """Load agent evaluation history from file."""
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load history: {e}[/yellow]")
        return []


def save_agent_history(history: list[dict[str, Any]]) -> None:
    """Save agent evaluation history to file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = history[-MAX_HISTORY_ENTRIES:]

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_to_agent_history(
    summary: E2EEvalSummary,
    run_id: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Add a new agent evaluation run to history."""
    history = load_agent_history()

    # Compute SLO pass/fail using eval-specific thresholds
    p95_slo_pass = summary.p95_latency_ms <= SLO_EVAL_LATENCY_P95_MS
    avg_slo_pass = summary.avg_latency_ms <= SLO_EVAL_LATENCY_AVG_MS

    # Compute which SLOs failed
    failed_slos = []
    if not p95_slo_pass:
        failed_slos.append(f"P95 latency {summary.p95_latency_ms:.0f}ms > {SLO_EVAL_LATENCY_P95_MS}ms")
    if not avg_slo_pass:
        failed_slos.append(f"Avg latency {summary.avg_latency_ms:.0f}ms > {SLO_EVAL_LATENCY_AVG_MS}ms")
    if summary.answer_relevance_rate < SLO_ANSWER_RELEVANCE:
        failed_slos.append(f"Answer relevance {summary.answer_relevance_rate:.1%} < {SLO_ANSWER_RELEVANCE:.0%}")
    if summary.groundedness_rate < SLO_GROUNDEDNESS:
        failed_slos.append(f"Groundedness {summary.groundedness_rate:.1%} < {SLO_GROUNDEDNESS:.0%}")
    if summary.mode_accuracy < SLO_MODE_ACCURACY:
        failed_slos.append(f"Mode accuracy {summary.mode_accuracy:.1%} < {SLO_MODE_ACCURACY:.0%}")

    entry = {
        "run_id": run_id or datetime.now().isoformat(),
        "timestamp": datetime.now().isoformat(),
        "tags": tags or [],
        "metrics": {
            "answer_relevance": summary.answer_relevance_rate,
            "groundedness": summary.groundedness_rate,
            "tool_selection": summary.tool_selection_accuracy,
            "mode_accuracy": summary.mode_accuracy,
            "p95_latency_ms": summary.p95_latency_ms,
            "avg_latency_ms": summary.avg_latency_ms,
        },
        "total_tests": summary.total_tests,
        "p95_slo_pass": p95_slo_pass,
        "avg_slo_pass": avg_slo_pass,
        "all_slos_passed": len(failed_slos) == 0,
        "failed_slos": failed_slos,
    }

    history.append(entry)
    save_agent_history(history)


# =============================================================================
# Trend Analysis
# =============================================================================

def compute_agent_trends(history: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Compute trend statistics for a metric."""
    if len(history) < 2:
        return {"has_trend": False}

    values = [h["metrics"].get(metric, 0) for h in history]

    if len(values) >= 6:
        recent_avg = sum(values[-3:]) / 3
        prev_avg = sum(values[-6:-3]) / 3
        trend = recent_avg - prev_avg
    elif len(values) >= 2:
        trend = values[-1] - values[-2]
    else:
        trend = 0

    return {
        "has_trend": True,
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "current": values[-1],
        "previous": values[-2] if len(values) >= 2 else None,
        "trend": trend,
        "trend_direction": "up" if trend > 0.01 else "down" if trend < -0.01 else "stable",
        "num_runs": len(values),
    }


# =============================================================================
# Reporting
# =============================================================================

def print_agent_trend_report(num_runs: int | None = None) -> None:
    """Print historical trend report for agent evaluation."""
    history = load_agent_history()

    if not history:
        console.print("[dim]No agent evaluation history found.[/dim]")
        return

    if num_runs:
        history = history[-num_runs:]

    console.print(Panel(
        f"[bold]Agent Evaluation History ({len(history)} runs)[/bold]",
        border_style="blue",
    ))

    # Trend summary
    metrics = [
        ("Answer Relevance", "answer_relevance", "%", True),
        ("Groundedness", "groundedness", "%", True),
        ("Mode Accuracy", "mode_accuracy", "%", True),
        ("Tool Selection", "tool_selection", "%", True),
        ("P95 Latency", "p95_latency_ms", "ms", False),
        ("Avg Latency", "avg_latency_ms", "ms", False),
    ]

    trend_table = Table(title="Agent Metric Trends", show_header=True, header_style="bold cyan")
    trend_table.add_column("Metric")
    trend_table.add_column("Current", justify="right")
    trend_table.add_column("Avg", justify="right")
    trend_table.add_column("Trend", justify="center")

    for label, metric, unit, higher_is_better in metrics:
        trend = compute_agent_trends(history, metric)

        if not trend.get("has_trend"):
            continue

        if unit == "%":
            curr_str = f"{trend['current']:.1%}"
            avg_str = f"{trend['avg']:.1%}"
        else:
            curr_str = f"{trend['current']:.0f}{unit}"
            avg_str = f"{trend['avg']:.0f}{unit}"

        direction = trend["trend_direction"]
        if direction == "up":
            trend_str = "[green]+[/green]" if higher_is_better else "[red]+[/red]"
        elif direction == "down":
            trend_str = "[red]-[/red]" if higher_is_better else "[green]-[/green]"
        else:
            trend_str = "[dim]=[/dim]"

        trend_table.add_row(label, curr_str, avg_str, trend_str)

    console.print(trend_table)

    # Recent runs
    runs_table = Table(title="Recent Agent Runs", show_header=True)
    runs_table.add_column("Date")
    runs_table.add_column("Rel", justify="right")
    runs_table.add_column("Gnd", justify="right")
    runs_table.add_column("Mode", justify="right")
    runs_table.add_column("P95", justify="right")
    runs_table.add_column("SLOs", justify="center")

    for entry in history[-5:]:
        try:
            dt = datetime.fromisoformat(entry["timestamp"])
            date_str = dt.strftime("%m-%d %H:%M")
        except Exception:
            date_str = entry["timestamp"][:10]

        rel = entry["metrics"].get("answer_relevance", 0)
        gnd = entry["metrics"].get("groundedness", 0)
        mode = entry["metrics"].get("mode_accuracy", 0)
        lat = entry["metrics"].get("p95_latency_ms", 0)
        all_passed = entry.get("all_slos_passed", False)
        failed_count = len(entry.get("failed_slos", []))

        slo_str = "[green]OK[/green]" if all_passed else f"[red]X{failed_count}[/red]"

        runs_table.add_row(
            date_str,
            f"{rel:.0%}",
            f"{gnd:.0%}",
            f"{mode:.0%}",
            f"{lat/1000:.1f}s",
            slo_str,
        )

    console.print(runs_table)
