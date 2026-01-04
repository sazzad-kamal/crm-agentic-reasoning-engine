"""
Agent Evaluation Tracking: Regression Detection & Latency Budget Monitoring.

Provides:
- Comparison against previous agent evaluation runs
- Latency budget violation tracking
- Rich console output for both

Mirrors backend.rag.eval.tracking for consistency.
"""

import json
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.panel import Panel

from backend.agent.eval.models import (
    E2EEvalResult,
    E2EEvalSummary,
    SLO_LATENCY_P95_MS,
)
from backend.agent.eval.base import console


# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
PREVIOUS_RESULTS_PATH = DATA_DIR / "agent_eval_results_previous.json"


# =============================================================================
# Latency Budgets for Agent Pipeline
# =============================================================================

AGENT_LATENCY_BUDGETS: dict[str, int] = {
    "router": 500,  # Intent routing LLM call
    "company_lookup": 200,  # Company data fetch
    "tool_execution": 300,  # Per-tool execution
    "rag_docs": 3000,  # RAG pipeline for docs
    "synthesis": 2000,  # Final answer synthesis
}

AGENT_TOTAL_LATENCY_BUDGET_MS = SLO_LATENCY_P95_MS


# =============================================================================
# Regression Detection
# =============================================================================


def load_previous_e2e_summary() -> E2EEvalSummary | None:
    """Load summary from previous E2E evaluation run."""
    if not PREVIOUS_RESULTS_PATH.exists():
        return None

    try:
        with open(PREVIOUS_RESULTS_PATH) as f:
            data = json.load(f)
            if "summary" in data:
                return E2EEvalSummary(**data["summary"])
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load previous results: {e}[/yellow]")
    return None


def save_e2e_as_previous(results: list[E2EEvalResult], summary: E2EEvalSummary) -> None:
    """Save current E2E results as previous for next comparison."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "results": [r.model_dump() for r in results],
        "summary": summary.model_dump(),
    }
    with open(PREVIOUS_RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def compare_e2e_with_previous(
    current: E2EEvalSummary,
    previous: E2EEvalSummary | None,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Compare current E2E results with previous run.

    Args:
        current: Current evaluation summary
        previous: Previous evaluation summary (or None)
        threshold: Minimum delta to flag as regression (default 5%)

    Returns:
        Dict with comparison details and regression flags
    """
    if previous is None:
        return {
            "has_previous": False,
            "regressions": [],
            "improvements": [],
        }

    metrics = [
        ("Answer Relevance", "answer_relevance_rate"),
        ("Groundedness", "groundedness_rate"),
    ]

    regressions = []
    improvements = []

    for label, attr in metrics:
        prev_val = getattr(previous, attr)
        curr_val = getattr(current, attr)
        delta = curr_val - prev_val

        if delta < -threshold:
            regressions.append(
                {
                    "metric": label,
                    "previous": prev_val,
                    "current": curr_val,
                    "delta": delta,
                }
            )
        elif delta > threshold:
            improvements.append(
                {
                    "metric": label,
                    "previous": prev_val,
                    "current": curr_val,
                    "delta": delta,
                }
            )

    # Latency comparison (inverse - increase is regression)
    latency_delta = current.p95_latency_ms - previous.p95_latency_ms
    latency_threshold = 500  # 500ms threshold for latency

    if latency_delta > latency_threshold:
        regressions.append(
            {
                "metric": "P95 Latency",
                "previous": previous.p95_latency_ms,
                "current": current.p95_latency_ms,
                "delta": latency_delta,
                "unit": "ms",
            }
        )
    elif latency_delta < -latency_threshold:
        improvements.append(
            {
                "metric": "P95 Latency",
                "previous": previous.p95_latency_ms,
                "current": current.p95_latency_ms,
                "delta": latency_delta,
                "unit": "ms",
            }
        )

    return {
        "has_previous": True,
        "regressions": regressions,
        "improvements": improvements,
        "previous_summary": previous,
        "current_summary": current,
    }


def print_e2e_regression_report(comparison: dict[str, Any]) -> None:
    """Print E2E regression comparison report."""
    if not comparison["has_previous"]:
        console.print(
            "[dim]No previous run to compare against. This run will be saved as baseline.[/dim]"
        )
        return

    previous = comparison["previous_summary"]
    current = comparison["current_summary"]

    # Build comparison table
    table = Table(
        title="E2E Quality Comparison (vs last run)", show_header=True, header_style="bold cyan"
    )
    table.add_column("Metric", style="dim")
    table.add_column("Previous", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")

    metrics = [
        ("Answer Relevance", "answer_relevance_rate", "%"),
        ("Groundedness", "groundedness_rate", "%"),
        ("P95 Latency", "p95_latency_ms", "ms"),
    ]

    for label, attr, unit in metrics:
        prev_val = getattr(previous, attr)
        curr_val = getattr(current, attr)
        delta = curr_val - prev_val

        # Format based on unit
        if unit == "%":
            prev_str = f"{prev_val:.1%}"
            curr_str = f"{curr_val:.1%}"
            delta_str = f"{delta:+.1%}"
        else:
            prev_str = f"{prev_val:.0f}{unit}"
            curr_str = f"{curr_val:.0f}{unit}"
            delta_str = f"{delta:+.0f}{unit}"

        # Color code delta
        if label == "P95 Latency":
            delta_color = "green" if delta < 0 else "red" if delta > 0 else "white"
        else:
            delta_color = "green" if delta > 0 else "red" if delta < 0 else "white"

        table.add_row(label, prev_str, curr_str, f"[{delta_color}]{delta_str}[/{delta_color}]")

    console.print(table)

    # Regression/improvement summary
    if comparison["regressions"]:
        console.print(
            f"\n[red bold][!] REGRESSION DETECTED: {len(comparison['regressions'])} metrics declined[/red bold]"
        )
        for r in comparison["regressions"]:
            if r.get("unit") == "ms":
                console.print(f"  - {r['metric']}: {r['delta']:+.0f}ms")
            else:
                console.print(f"  - {r['metric']}: {r['delta']:+.1%}")

    if comparison["improvements"]:
        console.print(
            f"\n[green bold][OK] IMPROVEMENT: {len(comparison['improvements'])} metrics improved[/green bold]"
        )


# =============================================================================
# Latency Budget Tracking
# =============================================================================


def analyze_e2e_budget_violations(
    results: list[E2EEvalResult],
) -> dict[str, Any]:
    """
    Analyze latency budget violations for E2E tests.

    Args:
        results: List of E2E evaluation results

    Returns:
        Dict with budget analysis
    """
    total_violations = []

    for result in results:
        if result.latency_ms > AGENT_TOTAL_LATENCY_BUDGET_MS:
            total_violations.append(
                {
                    "test_case_id": result.test_case_id,
                    "question": result.question[:50],
                    "latency_ms": result.latency_ms,
                    "over_by": result.latency_ms - AGENT_TOTAL_LATENCY_BUDGET_MS,
                }
            )

    # Group by category
    by_category: dict[str, list[float]] = {}
    for result in results:
        cat = result.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(result.latency_ms)

    category_stats = {}
    for cat, latencies in by_category.items():
        avg = sum(latencies) / len(latencies) if latencies else 0
        p95_idx = int(len(latencies) * 0.95)
        p95 = sorted(latencies)[min(p95_idx, len(latencies) - 1)] if latencies else 0
        exceeded = sum(1 for lat in latencies if lat > AGENT_TOTAL_LATENCY_BUDGET_MS)

        category_stats[cat] = {
            "count": len(latencies),
            "avg": avg,
            "p95": p95,
            "exceeded": exceeded,
        }

    return {
        "total_violations": total_violations,
        "category_stats": category_stats,
        "total_budget": AGENT_TOTAL_LATENCY_BUDGET_MS,
    }


def print_e2e_budget_report(results: list[E2EEvalResult]) -> None:
    """Print E2E latency budget report."""
    analysis = analyze_e2e_budget_violations(results)

    console.print("\n")

    # Category latency table
    if analysis["category_stats"]:
        table = Table(title="Latency by Category", show_header=True, header_style="bold cyan")
        table.add_column("Category", style="bold")
        table.add_column("Tests", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("Exceeded", justify="right")

        for cat, stats in sorted(analysis["category_stats"].items()):
            exceeded = stats["exceeded"]
            total = stats["count"]
            status = (
                "[green]OK[/green]" if exceeded == 0 else f"[yellow]! {exceeded}/{total}[/yellow]"
            )

            table.add_row(
                cat,
                str(total),
                f"{stats['avg']:.0f}ms",
                f"{stats['p95']:.0f}ms",
                status,
            )

        console.print(table)

    # Total latency violations
    if analysis["total_violations"]:
        console.print(
            f"\n[yellow bold][!] {len(analysis['total_violations'])} tests exceeded latency budget ({analysis['total_budget']}ms)[/yellow bold]"
        )

        for v in analysis["total_violations"][:5]:
            console.print(
                f"  - {v['test_case_id']}: {v['latency_ms']:.0f}ms (+{v['over_by']:.0f}ms over)"
            )
    else:
        console.print(
            f"\n[green][OK] All tests within latency budget ({analysis['total_budget']}ms)[/green]"
        )


# =============================================================================
# Combined Report
# =============================================================================


def print_e2e_tracking_report(
    results: list[E2EEvalResult],
    summary: E2EEvalSummary,
) -> None:
    """
    Print full E2E tracking report with regression detection and budget analysis.

    Args:
        results: Current E2E evaluation results
        summary: Current E2E evaluation summary
    """
    console.print(
        Panel(
            "[bold]Agent E2E Tracking Report[/bold]",
            border_style="blue",
        )
    )

    # 1. Regression detection
    previous = load_previous_e2e_summary()
    comparison = compare_e2e_with_previous(summary, previous)
    print_e2e_regression_report(comparison)

    # 2. Budget analysis
    print_e2e_budget_report(results)

    # 3. Add to history and show trends
    from backend.agent.eval.e2e.history import add_to_agent_history, print_agent_trend_report

    add_to_agent_history(summary)
    print_agent_trend_report(num_runs=5)

    # 4. Save current as previous
    save_e2e_as_previous(results, summary)
    console.print("\n[dim]Current results saved for next comparison.[/dim]")
