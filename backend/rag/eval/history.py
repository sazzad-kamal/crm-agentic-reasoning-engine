"""
Evaluation History Tracking.

Stores multiple evaluation runs and provides trend analysis:
- Track quality metrics over time
- Detect degradation trends
- Visualize improvement/regression patterns

Usage:
    from backend.rag.eval.history import add_to_history, print_trend_report
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.panel import Panel

from backend.rag.eval.models import DocsEvalSummary
from backend.rag.eval.base import console


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
HISTORY_FILE = DATA_DIR / "eval_history.json"
MAX_HISTORY_ENTRIES = 20  # Keep last 20 runs


# =============================================================================
# History Storage
# =============================================================================

def load_history() -> list[dict[str, Any]]:
    """Load evaluation history from file."""
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load history: {e}[/yellow]")
        return []


def save_history(history: list[dict[str, Any]]) -> None:
    """Save evaluation history to file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Keep only last N entries
    history = history[-MAX_HISTORY_ENTRIES:]

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_to_history(
    summary: DocsEvalSummary,
    run_id: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """
    Add a new evaluation run to history.

    Args:
        summary: Evaluation summary to store
        run_id: Optional identifier for this run (default: timestamp)
        tags: Optional tags for this run (e.g., ["baseline", "v2.0"])
    """
    history = load_history()

    entry = {
        "run_id": run_id or datetime.now().isoformat(),
        "timestamp": datetime.now().isoformat(),
        "tags": tags or [],
        "metrics": {
            "rag_triad": summary.rag_triad_success,
            "context_relevance": summary.context_relevance,
            "answer_relevance": summary.answer_relevance,
            "groundedness": summary.groundedness,
            "doc_recall": summary.avg_doc_recall,
            "p95_latency_ms": summary.p95_latency_ms,
            "avg_latency_ms": summary.avg_latency_ms,
            "total_tokens": summary.total_tokens,
        },
        "slos_passed": summary.all_slos_passed,
        "failed_slos": summary.failed_slos,
    }

    history.append(entry)
    save_history(history)


# =============================================================================
# Trend Analysis
# =============================================================================

def compute_trends(history: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """
    Compute trend statistics for a metric.

    Returns:
        Dict with min, max, avg, current, trend direction
    """
    if len(history) < 2:
        return {"has_trend": False}

    values = [h["metrics"].get(metric, 0) for h in history]

    # Recent trend (last 3 vs previous 3)
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


def detect_degradation(history: list[dict[str, Any]], threshold: float = 0.05) -> list[str]:
    """
    Detect metrics that are degrading over time.

    Returns:
        List of metric names showing degradation
    """
    degrading = []

    quality_metrics = ["rag_triad", "context_relevance", "answer_relevance", "groundedness", "doc_recall"]

    for metric in quality_metrics:
        trend = compute_trends(history, metric)
        if trend.get("has_trend") and trend.get("trend", 0) < -threshold:
            degrading.append(metric)

    # For latency, increasing is bad
    latency_trend = compute_trends(history, "p95_latency_ms")
    if latency_trend.get("has_trend") and latency_trend.get("trend", 0) > 500:  # 500ms increase
        degrading.append("p95_latency_ms")

    return degrading


# =============================================================================
# Reporting
# =============================================================================

def print_trend_report(num_runs: int | None = None) -> None:
    """
    Print historical trend report.

    Args:
        num_runs: Number of recent runs to show (default: all)
    """
    history = load_history()

    if not history:
        console.print("[dim]No evaluation history found.[/dim]")
        return

    if num_runs:
        history = history[-num_runs:]

    console.print(Panel(
        f"[bold]Evaluation History ({len(history)} runs)[/bold]",
        border_style="blue",
    ))

    # Trend summary table
    metrics = [
        ("RAG Triad", "rag_triad", "%", True),
        ("Context Rel.", "context_relevance", "%", True),
        ("Answer Rel.", "answer_relevance", "%", True),
        ("Groundedness", "groundedness", "%", True),
        ("Doc Recall", "doc_recall", "%", True),
        ("P95 Latency", "p95_latency_ms", "ms", False),
        ("Avg Latency", "avg_latency_ms", "ms", False),
    ]

    trend_table = Table(title="Metric Trends", show_header=True, header_style="bold cyan")
    trend_table.add_column("Metric")
    trend_table.add_column("Current", justify="right")
    trend_table.add_column("Avg", justify="right")
    trend_table.add_column("Min", justify="right")
    trend_table.add_column("Max", justify="right")
    trend_table.add_column("Trend", justify="center")

    for label, metric, unit, higher_is_better in metrics:
        trend = compute_trends(history, metric)

        if not trend.get("has_trend"):
            continue

        # Format values
        if unit == "%":
            curr_str = f"{trend['current']:.1%}"
            avg_str = f"{trend['avg']:.1%}"
            min_str = f"{trend['min']:.1%}"
            max_str = f"{trend['max']:.1%}"
        else:
            curr_str = f"{trend['current']:.0f}{unit}"
            avg_str = f"{trend['avg']:.0f}{unit}"
            min_str = f"{trend['min']:.0f}{unit}"
            max_str = f"{trend['max']:.0f}{unit}"

        # Trend indicator
        direction = trend["trend_direction"]
        if direction == "up":
            if higher_is_better:
                trend_str = "[green]↑ improving[/green]"
            else:
                trend_str = "[red]↑ degrading[/red]"
        elif direction == "down":
            if higher_is_better:
                trend_str = "[red]↓ degrading[/red]"
            else:
                trend_str = "[green]↓ improving[/green]"
        else:
            trend_str = "[dim]→ stable[/dim]"

        trend_table.add_row(label, curr_str, avg_str, min_str, max_str, trend_str)

    console.print(trend_table)

    # Recent runs table
    console.print("\n")
    runs_table = Table(title="Recent Runs", show_header=True, header_style="bold")
    runs_table.add_column("Run", style="dim")
    runs_table.add_column("Date")
    runs_table.add_column("RAG Triad", justify="right")
    runs_table.add_column("Latency", justify="right")
    runs_table.add_column("SLOs", justify="center")
    runs_table.add_column("Tags")

    for entry in history[-10:]:  # Last 10 runs
        run_id = entry["run_id"][:16] if len(entry["run_id"]) > 16 else entry["run_id"]

        try:
            dt = datetime.fromisoformat(entry["timestamp"])
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_str = entry["timestamp"][:16]

        triad = entry["metrics"].get("rag_triad", 0)
        latency = entry["metrics"].get("p95_latency_ms", 0)
        slos_passed = entry.get("slos_passed", True)
        failed_slos = entry.get("failed_slos", [])
        failed_count = len(failed_slos)
        tags = ", ".join(entry.get("tags", []))

        triad_color = "green" if triad >= 0.7 else "yellow" if triad >= 0.5 else "red"
        slo_str = "[green]✓[/green]" if slos_passed else f"[red]✗{failed_count}[/red]"

        runs_table.add_row(
            run_id,
            date_str,
            f"[{triad_color}]{triad:.1%}[/{triad_color}]",
            f"{latency:.0f}ms",
            slo_str,
            tags or "-",
        )

    console.print(runs_table)

    # Degradation warnings
    degrading = detect_degradation(history)
    if degrading:
        console.print(f"\n[yellow bold]⚠ Degradation detected in: {', '.join(degrading)}[/yellow bold]")
        console.print("[dim]Consider investigating recent changes.[/dim]")
