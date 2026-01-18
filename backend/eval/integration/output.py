"""Output and display functions for evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.table import Table

from backend.eval.integration.models import (
    SLO_FLOW_ANSWER_CORRECTNESS,
    SLO_FLOW_AVG_LATENCY_MS,
    SLO_FLOW_COMPOSITE_SCORE,
    SLO_FLOW_FAITHFULNESS,
    SLO_FLOW_PATH_PASS_RATE,
    SLO_FLOW_QUESTION_PASS_RATE,
    SLO_FLOW_RELEVANCE,
    FlowEvalResults,
    FlowStepResult,
)
from backend.eval.shared.formatting import build_eval_table, console, format_percentage

logger = logging.getLogger(__name__)


def print_summary(results: FlowEvalResults, latency_pcts: dict[str, float] | None = None) -> bool:
    """
    Print a comprehensive summary of eval results with SLO status.

    Args:
        results: The evaluation results
        latency_pcts: Optional latency breakdown from LangSmith (informational only)

    Returns:
        True if all SLOs passed
    """
    console.print()

    # Compute SLO pass/fail
    path_slo_pass = results.path_pass_rate >= SLO_FLOW_PATH_PASS_RATE
    q_slo_pass = results.question_pass_rate >= SLO_FLOW_QUESTION_PASS_RATE
    relevance_slo_pass = results.avg_relevance >= SLO_FLOW_RELEVANCE
    faithfulness_slo_pass = results.avg_faithfulness >= SLO_FLOW_FAITHFULNESS
    answer_correctness_slo_pass = results.avg_answer_correctness >= SLO_FLOW_ANSWER_CORRECTNESS
    avg_latency_pass = results.avg_latency_per_question_ms <= SLO_FLOW_AVG_LATENCY_MS
    composite_pass = results.composite_score >= SLO_FLOW_COMPOSITE_SCORE

    # Build table sections: (section_name, [(label, value, slo_target, slo_passed)])
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]] = [
        (
            "Answer Quality",
            [
                (
                    "  Relevance",
                    format_percentage(results.avg_relevance),
                    f">={format_percentage(SLO_FLOW_RELEVANCE)}",
                    relevance_slo_pass,
                ),
                (
                    "  Faithfulness",
                    format_percentage(results.avg_faithfulness),
                    f">={format_percentage(SLO_FLOW_FAITHFULNESS)}",
                    faithfulness_slo_pass,
                ),
                (
                    "  Answer Correctness",
                    format_percentage(results.avg_answer_correctness),
                    f">={format_percentage(SLO_FLOW_ANSWER_CORRECTNESS)}",
                    answer_correctness_slo_pass,
                ),
            ],
        ),
        (
            "Latency",
            [
                (
                    "  avg/question",
                    f"{results.avg_latency_per_question_ms:.0f}ms",
                    f"<={SLO_FLOW_AVG_LATENCY_MS}ms",
                    avg_latency_pass,
                ),
            ],
        ),
        (
            "RAGAS Reliability",
            [
                (
                    "  Metrics Success",
                    f"{results.ragas_metrics_total - results.ragas_metrics_failed}/{results.ragas_metrics_total} ({format_percentage(results.ragas_success_rate)})",
                    ">=90.0%",
                    results.ragas_success_rate >= 0.9,
                ),
            ],
        ),
    ]

    # Optional LangSmith section (informational only, no SLO)
    if latency_pcts:
        sections.append((
            "LangSmith (info)",
            [
                ("  Fetch", format_percentage(latency_pcts.get("fetch", 0)), "-", None),
                ("  Answer", format_percentage(latency_pcts.get("answer", 0)), "-", None),
                ("  Followup", format_percentage(latency_pcts.get("followup", 0)), "-", None),
            ],
        ))

    # Build table with composite score as aggregate row
    summary_table = build_eval_table(
        "Flow Evaluation Summary",
        sections,
        aggregate_row=(
            "Composite Score",
            format_percentage(results.composite_score),
            f">={format_percentage(SLO_FLOW_COMPOSITE_SCORE)}",
            composite_pass,
        ),
    )
    console.print(summary_table)

    # All SLOs must pass
    all_slos_passed = (
        path_slo_pass
        and q_slo_pass
        and relevance_slo_pass
        and faithfulness_slo_pass
        and answer_correctness_slo_pass
        and avg_latency_pass
        and composite_pass
    )

    # SLO Failures Detail
    _print_slo_failures(results)

    return all_slos_passed


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
    """Print details of SLO failures as a compact table."""
    # Collect all failed steps with their path info
    failures: list[tuple[int, FlowStepResult]] = []
    for flow_result in results.all_results:
        for step in flow_result.steps:
            if _count_slo_failures(step) > 0:
                failures.append((flow_result.path_id, step))

    if not failures:
        return

    # Sort by failure count (most failures first)
    failures.sort(key=lambda x: _count_slo_failures(x[1]), reverse=True)

    # Show top 5
    shown = failures[:5]
    total = len(failures)

    console.print()
    failed_table = Table(
        title=f"SLO Failures ({len(shown)} of {total} shown, sorted by severity)",
        show_header=True,
        header_style="bold yellow",
    )
    failed_table.add_column("Path", style="dim", width=4)
    failed_table.add_column("Question", width=40)
    failed_table.add_column("R", justify="center", width=3)
    failed_table.add_column("F", justify="center", width=3)
    failed_table.add_column("A", justify="center", width=3)

    def fmt(passed: bool) -> str:
        return "[green]Y[/green]" if passed else "[red]X[/red]"

    for path_id, step in shown:
        question_display = step.question[:38] + "..." if len(step.question) > 38 else step.question

        r_pass = step.relevance_score >= SLO_FLOW_RELEVANCE
        f_pass = step.faithfulness_score >= SLO_FLOW_FAITHFULNESS
        a_pass = step.answer_correctness_score >= SLO_FLOW_ANSWER_CORRECTNESS

        failed_table.add_row(
            str(path_id + 1),
            question_display,
            fmt(r_pass),
            fmt(f_pass),
            fmt(a_pass),
        )

    console.print(failed_table)


def save_results(results: FlowEvalResults, output_path: Path) -> None:
    """Save results to JSON file."""
    data = {
        "summary": {
            "composite_score": results.composite_score,
            "total_paths": results.total_paths,
            "paths_tested": results.paths_tested,
            "paths_passed": results.paths_passed,
            "paths_failed": results.paths_failed,
            "path_pass_rate": results.path_pass_rate,
            "total_questions": results.total_questions,
            "questions_passed": results.questions_passed,
            "questions_failed": results.questions_failed,
            "question_pass_rate": results.question_pass_rate,
            "avg_relevance": results.avg_relevance,
            "avg_faithfulness": results.avg_faithfulness,
            "avg_answer_correctness": results.avg_answer_correctness,
            "total_latency_ms": results.total_latency_ms,
            "avg_latency_per_question_ms": results.avg_latency_per_question_ms,
            "wall_clock_ms": results.wall_clock_ms,
            "ragas_metrics_total": results.ragas_metrics_total,
            "ragas_metrics_failed": results.ragas_metrics_failed,
            "ragas_success_rate": results.ragas_success_rate,
        },
        "slo_results": {
            "path_pass_rate": {
                "value": results.path_pass_rate,
                "target": SLO_FLOW_PATH_PASS_RATE,
                "passed": results.path_pass_rate >= SLO_FLOW_PATH_PASS_RATE,
            },
            "question_pass_rate": {
                "value": results.question_pass_rate,
                "target": SLO_FLOW_QUESTION_PASS_RATE,
                "passed": results.question_pass_rate >= SLO_FLOW_QUESTION_PASS_RATE,
            },
            "relevance": {
                "value": results.avg_relevance,
                "target": SLO_FLOW_RELEVANCE,
                "passed": results.avg_relevance >= SLO_FLOW_RELEVANCE,
            },
            "faithfulness": {
                "value": results.avg_faithfulness,
                "target": SLO_FLOW_FAITHFULNESS,
                "passed": results.avg_faithfulness >= SLO_FLOW_FAITHFULNESS,
            },
            "answer_correctness": {
                "value": results.avg_answer_correctness,
                "target": SLO_FLOW_ANSWER_CORRECTNESS,
                "passed": results.avg_answer_correctness >= SLO_FLOW_ANSWER_CORRECTNESS,
            },
            "avg_latency_ms": {
                "value": results.avg_latency_per_question_ms,
                "target": SLO_FLOW_AVG_LATENCY_MS,
                "passed": results.avg_latency_per_question_ms <= SLO_FLOW_AVG_LATENCY_MS,
            },
            "composite_score": {
                "value": results.composite_score,
                "target": SLO_FLOW_COMPOSITE_SCORE,
                "passed": results.composite_score >= SLO_FLOW_COMPOSITE_SCORE,
            },
        },
        "failed_paths": [
            {
                "path_id": fp.path_id,
                "questions": fp.questions,
                "steps": [
                    {
                        "question": s.question,
                        "has_answer": s.has_answer,
                        "relevance_score": s.relevance_score,
                        "faithfulness_score": s.faithfulness_score,
                        "answer_correctness_score": s.answer_correctness_score,
                        "latency_ms": s.latency_ms,
                        "judge_explanation": s.judge_explanation,
                        "error": s.error,
                    }
                    for s in fp.steps
                ],
            }
            for fp in results.failed_paths
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[dim]Results saved to {output_path}[/dim]")


def check_qdrant_access() -> bool:
    """
    Check if Qdrant storage is accessible (not locked by another process).

    Returns:
        True if accessible, False if locked
    """
    try:
        from backend.agent.fetch.rag.client import get_qdrant_client

        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        if "already accessed" in str(e).lower():
            return False
        logger.warning(f"Qdrant check error: {e}")
        return True
