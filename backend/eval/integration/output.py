"""Output and display functions for evaluation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from backend.eval.integration.models import (
    SLO_FLOW_ANSWER_CORRECTNESS,
    SLO_FLOW_AVG_LATENCY_MS,
    SLO_FLOW_FAITHFULNESS,
    SLO_FLOW_PATH_PASS_RATE,
    SLO_FLOW_QUESTION_PASS_RATE,
    SLO_FLOW_RELEVANCE,
    FlowEvalResults,
    FlowStepResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data-driven SLO definitions
# =============================================================================


class SloSpec(NamedTuple):
    """Single SLO metric specification."""

    key: str  # JSON key for save_results
    label: str  # Display label
    section: str  # Grouping header
    get_value: Callable[[FlowEvalResults], float]
    target: float
    compare: str  # ">=" or "<="
    fmt: str  # "pct" or "ms"


SLO_SPECS: list[SloSpec] = [
    SloSpec("path_pass_rate", "Path Pass Rate", "Pass Rates",
            lambda r: r.path_pass_rate, SLO_FLOW_PATH_PASS_RATE, ">=", "pct"),
    SloSpec("question_pass_rate", "Question Pass Rate", "Pass Rates",
            lambda r: r.question_pass_rate, SLO_FLOW_QUESTION_PASS_RATE, ">=", "pct"),
    SloSpec("relevance", "Relevance", "Answer Quality",
            lambda r: r.avg_relevance, SLO_FLOW_RELEVANCE, ">=", "pct"),
    SloSpec("faithfulness", "Faithfulness", "Answer Quality",
            lambda r: r.avg_faithfulness, SLO_FLOW_FAITHFULNESS, ">=", "pct"),
    SloSpec("answer_correctness", "Answer Correctness", "Answer Quality",
            lambda r: r.avg_answer_correctness, SLO_FLOW_ANSWER_CORRECTNESS, ">=", "pct"),
    SloSpec("avg_latency_ms", "Avg Latency/Question", "Latency",
            lambda r: r.avg_latency_per_question_ms, SLO_FLOW_AVG_LATENCY_MS, "<=", "ms"),
]


def _slo_passed(spec: SloSpec, results: FlowEvalResults) -> bool:
    """Check if an SLO spec passes for the given results."""
    value = spec.get_value(results)
    if spec.compare == ">=":
        return value >= spec.target
    return value <= spec.target


def _format_slo(spec: SloSpec, value: float) -> tuple[str, str]:
    """Format a value and its SLO target for display."""
    if spec.fmt == "pct":
        return f"{value:.1%}", f"{spec.compare}{spec.target:.1%}"
    return f"{value:.0f}ms", f"{spec.compare}{spec.target:.0f}ms"


# =============================================================================
# Display functions
# =============================================================================


def print_summary(results: FlowEvalResults, latency_pcts: dict[str, float] | None = None) -> bool:
    """
    Print evaluation summary with SLO status.

    Returns:
        True if all SLOs passed.
    """
    print()
    print("Flow Evaluation Summary")
    print("=" * 50)

    all_passed = True
    current_section = ""

    for spec in SLO_SPECS:
        if spec.section != current_section:
            current_section = spec.section
            print(f"\n{current_section}")

        value = spec.get_value(results)
        passed = _slo_passed(spec, results)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        val_str, target_str = _format_slo(spec, value)
        print(f"  {spec.label}: {val_str} ({target_str} SLO) {status}")

    # RAGAS Reliability (special: ratio display, not in SLO_SPECS)
    ragas_ok = results.ragas_metrics_total - results.ragas_metrics_failed
    ragas_passed = results.ragas_success_rate >= 0.9
    if not ragas_passed:
        all_passed = False
    print("\nRAGAS Reliability")
    print(
        f"  Metrics Success: {ragas_ok}/{results.ragas_metrics_total}"
        f" ({results.ragas_success_rate:.1%}) (>=90.0% SLO)"
        f" {'PASS' if ragas_passed else 'FAIL'}"
    )

    # Optional LangSmith info (no SLO)
    if latency_pcts:
        print("\nLangSmith (info)")
        for key in ("fetch", "answer", "followup"):
            print(f"  {key.capitalize()}: {latency_pcts.get(key, 0):.1%}")

    # SLO Failures Detail
    _print_slo_failures(results)

    return all_passed


def _count_slo_failures(step: FlowStepResult) -> int:
    """Count how many SLO metrics failed for a step."""
    count = 0
    if step.relevance_score < SLO_FLOW_RELEVANCE:
        count += 1
    if step.faithfulness_score < SLO_FLOW_FAITHFULNESS:
        count += 1
    if step.answer_correctness_score < SLO_FLOW_ANSWER_CORRECTNESS:
        count += 1
    return count


def _print_slo_failures(results: FlowEvalResults) -> None:
    """Print details of SLO failures."""
    failures: list[tuple[int, FlowStepResult]] = []
    for flow_result in results.all_results:
        for step in flow_result.steps:
            if _count_slo_failures(step) > 0:
                failures.append((flow_result.path_id, step))

    if not failures:
        return

    failures.sort(key=lambda x: _count_slo_failures(x[1]), reverse=True)
    shown = failures[:5]

    print()
    print(f"SLO Failures ({len(shown)} of {len(failures)} shown, sorted by severity)")
    print(f"  {'Path':<5} {'Question':<40} {'R':>3} {'F':>3} {'A':>3}")
    print(f"  {'-'*5} {'-'*40} {'-'*3} {'-'*3} {'-'*3}")

    def fmt(passed: bool) -> str:
        return "Y" if passed else "X"

    for path_id, step in shown:
        q = step.question[:38] + "..." if len(step.question) > 38 else step.question
        r = fmt(step.relevance_score >= SLO_FLOW_RELEVANCE)
        f = fmt(step.faithfulness_score >= SLO_FLOW_FAITHFULNESS)
        a = fmt(step.answer_correctness_score >= SLO_FLOW_ANSWER_CORRECTNESS)
        print(f"  {path_id+1:<5} {q:<40} {r:>3} {f:>3} {a:>3}")


# =============================================================================
# JSON export
# =============================================================================


def save_results(results: FlowEvalResults, output_path: Path) -> None:
    """Save results to JSON file."""
    summary = results.model_dump(exclude={"failed_paths", "all_results"})

    slo_results = {}
    for spec in SLO_SPECS:
        value = spec.get_value(results)
        slo_results[spec.key] = {
            "value": value,
            "target": spec.target,
            "passed": _slo_passed(spec, results),
        }

    data = {
        "summary": summary,
        "slo_results": slo_results,
        "failed_paths": [fp.model_dump() for fp in results.failed_paths],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output_path}")
