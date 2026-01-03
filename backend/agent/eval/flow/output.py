"""Output and display functions for flow evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.table import Table

from backend.agent.eval.base import console, format_percentage, format_check_mark
from backend.agent.eval.shared import (
    print_slo_result,
    get_failed_slos,
    print_overall_result_panel,
    build_eval_table,
)
from backend.agent.eval.models import (
    SLO_FLOW_PATH_PASS_RATE,
    SLO_FLOW_QUESTION_PASS_RATE,
    SLO_FLOW_RELEVANCE,
    SLO_FLOW_GROUNDED,
    FlowEvalResults,
)

logger = logging.getLogger(__name__)


def print_summary(results: FlowEvalResults) -> bool:
    """
    Print a comprehensive summary of eval results with SLO status.

    Returns:
        True if all SLOs passed
    """
    console.print()

    # Compute SLO pass/fail
    path_slo_pass = results.path_pass_rate >= SLO_FLOW_PATH_PASS_RATE
    q_slo_pass = results.question_pass_rate >= SLO_FLOW_QUESTION_PASS_RATE
    relevance_slo_pass = results.avg_relevance >= SLO_FLOW_RELEVANCE
    grounded_slo_pass = results.avg_grounded >= SLO_FLOW_GROUNDED

    # Wall clock time
    wall_secs = results.wall_clock_ms / 1000

    # Build table sections: (section_name, [(label, value, slo_target, slo_passed)])
    sections = [
        (
            "",
            [
                ("Paths Tested", f"{results.paths_tested}/{results.total_paths}", None, None),
                (
                    "Path Pass Rate",
                    format_percentage(results.path_pass_rate),
                    f">={format_percentage(SLO_FLOW_PATH_PASS_RATE)}",
                    path_slo_pass,
                ),
            ],
        ),
        (
            "",
            [
                ("Questions Total", str(results.total_questions), None, None),
                (
                    "Question Pass Rate",
                    format_percentage(results.question_pass_rate),
                    f">={format_percentage(SLO_FLOW_QUESTION_PASS_RATE)}",
                    q_slo_pass,
                ),
            ],
        ),
        (
            "LLM Judge Scores",
            [
                (
                    "  Relevance",
                    format_percentage(results.avg_relevance),
                    f">={format_percentage(SLO_FLOW_RELEVANCE)}",
                    relevance_slo_pass,
                ),
                (
                    "  Groundedness",
                    format_percentage(results.avg_grounded),
                    f">={format_percentage(SLO_FLOW_GROUNDED)}",
                    grounded_slo_pass,
                ),
            ],
        ),
        (
            "Latency",
            [
                ("  Avg per Question", f"{results.avg_latency_per_question_ms:.0f}ms", None, None),
                ("  P95 per Question", f"{results.p95_latency_ms:.0f}ms", None, None),
                ("  Total", f"{results.total_latency_ms}ms", None, None),
            ],
        ),
        (
            "",
            [("Wall Clock Time", f"{wall_secs:.1f}s ({wall_secs / 60:.1f} min)", None, None)],
        ),
    ]

    summary_table = build_eval_table("Flow Evaluation Summary", sections)
    console.print(summary_table)

    # SLO Summary Panel
    slo_checks = [
        (
            "Path Pass Rate",
            path_slo_pass,
            format_percentage(path_pass_rate),
            f">={format_percentage(SLO_FLOW_PATH_PASS_RATE)}",
        ),
        (
            "Question Pass Rate",
            q_slo_pass,
            format_percentage(q_pass_rate),
            f">={format_percentage(SLO_FLOW_QUESTION_PASS_RATE)}",
        ),
        (
            "Relevance",
            relevance_slo_pass,
            format_percentage(results.avg_relevance),
            f">={format_percentage(SLO_FLOW_RELEVANCE)}",
        ),
        (
            "Groundedness",
            grounded_slo_pass,
            format_percentage(results.avg_grounded),
            f">={format_percentage(SLO_FLOW_GROUNDED)}",
        ),
    ]

    all_slos_passed = print_slo_result(slo_checks)

    # Failed Paths Detail
    if results.failed_paths:
        console.print()
        failed_table = Table(
            title=f"Failed Paths ({len(results.failed_paths)} total, showing first 5)",
            show_header=True,
            header_style="bold yellow",
        )
        failed_table.add_column("Path", style="bold", width=6)
        failed_table.add_column("Question", width=50)
        failed_table.add_column("R", justify="center", width=3)
        failed_table.add_column("G", justify="center", width=3)
        failed_table.add_column("Latency", justify="right", width=8)
        failed_table.add_column("Issue", width=40)

        for fp in results.failed_paths[:5]:
            for i, step in enumerate(fp.steps):
                if not step.passed:
                    issue = (
                        step.judge_explanation[:40]
                        if step.judge_explanation
                        else (step.error or "Unknown")
                    )
                    failed_table.add_row(
                        str(fp.path_id) if i == 0 else "",
                        step.question[:48] + "..." if len(step.question) > 48 else step.question,
                        format_check_mark(step.relevance_score == 1),
                        format_check_mark(step.grounded_score == 1),
                        f"{step.latency_ms}ms",
                        issue,
                    )

        console.print(failed_table)

    # Overall Result
    failed_slo_names = get_failed_slos(slo_checks)
    all_passed = all_slos_passed and results.paths_failed == 0

    failure_reasons = []
    if results.paths_failed > 0:
        failure_reasons.append(f"{results.paths_failed} paths failed")
    if failed_slo_names:
        failure_reasons.append(f"{len(failed_slo_names)} SLOs not met: {', '.join(failed_slo_names)}")

    console.print()
    print_overall_result_panel(
        all_passed=all_passed,
        failure_reasons=failure_reasons,
        success_message=f"All {results.paths_tested} paths passed, all SLOs met",
    )

    return all_slos_passed


def save_results(results: FlowEvalResults, output_path: Path) -> None:
    """Save results to JSON file."""
    data = {
        "summary": {
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
            "avg_grounded": results.avg_grounded,
            "total_latency_ms": results.total_latency_ms,
            "avg_latency_per_question_ms": results.avg_latency_per_question_ms,
            "p95_latency_ms": results.p95_latency_ms,
            "wall_clock_ms": results.wall_clock_ms,
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
            "groundedness": {
                "value": results.avg_grounded,
                "target": SLO_FLOW_GROUNDED,
                "passed": results.avg_grounded >= SLO_FLOW_GROUNDED,
            },
        },
        "tracked_metrics": {
            "avg_latency_ms": results.avg_latency_per_question_ms,
            "p95_latency_ms": results.p95_latency_ms,
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
                        "grounded_score": s.grounded_score,
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
        from backend.agent.rag.client import get_qdrant_client

        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        if "already accessed" in str(e).lower():
            return False
        logger.warning(f"Qdrant check error: {e}")
        return True
