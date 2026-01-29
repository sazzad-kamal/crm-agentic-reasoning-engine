"""Output and display functions for evaluation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from backend.eval.answer.text.models import SLO_TEXT_ANSWER_CORRECTNESS, SLO_TEXT_ANSWER_RELEVANCY
from backend.eval.integration.models import (
    SLO_CONVO_STEP_PASS_RATE,
    ConvoEvalResults,
    ConvoStepResult,
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
    get_value: Callable[[ConvoEvalResults], float]
    target: float
    compare: str  # ">=" or "<="
    fmt: str  # "pct" or "ms"


SLO_SPECS: list[SloSpec] = [
    SloSpec("pass_rate", "Pass Rate", "Pass Rates",
            lambda r: r.pass_rate, SLO_CONVO_STEP_PASS_RATE, ">=", "pct"),
    SloSpec("relevance", "Relevance", "Answer Quality",
            lambda r: r.avg_relevance, SLO_TEXT_ANSWER_RELEVANCY, ">=", "pct"),
    SloSpec("answer_correctness", "Answer Correctness", "Answer Quality",
            lambda r: r.avg_answer_correctness, SLO_TEXT_ANSWER_CORRECTNESS, ">=", "pct"),
]


def _slo_passed(spec: SloSpec, results: ConvoEvalResults) -> bool:
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


def print_summary(results: ConvoEvalResults, latency_pcts: dict[str, float] | None = None) -> bool:
    """
    Print evaluation summary with SLO status.

    Returns:
        True if all SLOs passed.
    """
    print()
    print("Conversation Evaluation Summary")
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


def _count_slo_failures(step: ConvoStepResult) -> int:
    """Count how many SLO metrics failed for a step."""
    count = 0
    if step.relevance_score < SLO_TEXT_ANSWER_RELEVANCY:
        count += 1
    if step.answer_correctness_score < SLO_TEXT_ANSWER_CORRECTNESS:
        count += 1
    return count


def _print_slo_failures(results: ConvoEvalResults) -> None:
    """Print details of SLO failures."""
    failures: list[ConvoStepResult] = [
        c for c in results.cases if _count_slo_failures(c) > 0
    ]

    if not failures:
        return

    failures.sort(key=lambda x: _count_slo_failures(x), reverse=True)
    shown = failures[:5]

    print()
    print(f"SLO Failures ({len(shown)} of {len(failures)} shown, sorted by severity)")
    print(f"  {'Question':<45} {'R':>3} {'A':>3}")
    print(f"  {'-'*45} {'-'*3} {'-'*3}")

    def fmt(passed: bool) -> str:
        return "Y" if passed else "X"

    for step in shown:
        q = step.question[:43] + "..." if len(step.question) > 43 else step.question
        r = fmt(step.relevance_score >= SLO_TEXT_ANSWER_RELEVANCY)
        a = fmt(step.answer_correctness_score >= SLO_TEXT_ANSWER_CORRECTNESS)
        print(f"  {q:<45} {r:>3} {a:>3}")


# =============================================================================
# JSON export
# =============================================================================


def save_results(results: ConvoEvalResults, output_path: Path) -> None:
    """Save results to JSON file."""
    summary = results.model_dump(exclude={"cases"})

    slo_results = {}
    for spec in SLO_SPECS:
        value = spec.get_value(results)
        slo_results[spec.key] = {
            "value": value,
            "target": spec.target,
            "passed": _slo_passed(spec, results),
        }

    failed_cases = [c.model_dump() for c in results.cases if not c.passed]

    data = {
        "summary": summary,
        "slo_results": slo_results,
        "failed_cases": failed_cases,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output_path}")
