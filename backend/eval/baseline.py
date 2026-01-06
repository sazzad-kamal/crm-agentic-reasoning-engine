"""
Baseline comparison utilities for eval.

Handles saving, loading, and comparing evaluation baselines.
"""

import json
from pathlib import Path
from typing import Any

from backend.eval.formatting import console

REGRESSION_THRESHOLD = 0.05  # 5% regression threshold


def compare_to_baseline(
    current_score: float,
    baseline_path: Path,
    score_key: str = "overall_score",
) -> tuple[bool, float | None]:
    """
    Compare current score to a baseline.

    Args:
        current_score: Current evaluation score
        baseline_path: Path to baseline JSON file
        score_key: Key in baseline JSON containing the score

    Returns:
        Tuple of (is_regression, baseline_score)
    """
    if not baseline_path.exists():
        return False, None

    try:
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        # Handle nested summary structure
        if "summary" in baseline_data:
            baseline_score = baseline_data["summary"].get(score_key, 0.0)
        else:
            baseline_score = baseline_data.get(score_key, 0.0)

        # Regression if we're more than threshold worse
        is_regression = current_score < (baseline_score - REGRESSION_THRESHOLD)

        return is_regression, baseline_score
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load baseline: {e}[/yellow]")
        return False, None


def save_baseline(
    summary: dict[str, Any],
    baseline_path: Path,
) -> None:
    """
    Save current results as new baseline.

    Args:
        summary: Summary dictionary to save
        baseline_path: Path to save baseline JSON
    """
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump({"summary": summary}, f, indent=2, default=str)
    console.print(f"[green]Baseline saved to {baseline_path}[/green]")


def print_baseline_comparison(
    current_score: float,
    baseline_score: float | None,
    is_regression: bool,
) -> None:
    """
    Print baseline comparison results.

    Args:
        current_score: Current evaluation score
        baseline_score: Baseline score (or None if no baseline)
        is_regression: Whether regression was detected
    """
    if baseline_score is None:
        console.print("[dim]No baseline found for comparison[/dim]")
        return

    diff = current_score - baseline_score
    diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"

    if is_regression:
        console.print("\n[red bold]REGRESSION DETECTED[/red bold]")
        console.print(
            f"  Baseline: {baseline_score:.1%} -> Current: {current_score:.1%} ({diff_str})"
        )
    else:
        color = "green" if diff >= 0 else "yellow"
        console.print(
            f"\n[dim]Baseline: {baseline_score:.1%} -> Current: {current_score:.1%} ([{color}]{diff_str}[/{color}])[/dim]"
        )


__all__ = [
    "REGRESSION_THRESHOLD",
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
]
