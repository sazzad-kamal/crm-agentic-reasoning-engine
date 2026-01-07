"""Output and display functions for flow evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.table import Table

from backend.eval.base import console, format_percentage
from backend.eval.formatting import build_eval_table
from backend.eval.models import (
    SLO_COMPANY_EXTRACTION,
    SLO_FLOW_ANSWER_CORRECTNESS,
    SLO_FLOW_CONTEXT_PRECISION,
    SLO_FLOW_FAITHFULNESS,
    SLO_FLOW_PATH_PASS_RATE,
    SLO_FLOW_QUESTION_PASS_RATE,
    SLO_FLOW_RELEVANCE,
    SLO_LATENCY_ANSWER_PCT,
    SLO_LATENCY_RETRIEVAL_PCT,
    SLO_LATENCY_ROUTING_PCT,
    SLO_ROUTER_ACCURACY,
    FlowEvalResults,
    FlowStepResult,
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
    company_slo_pass = results.company_extraction_accuracy >= SLO_COMPANY_EXTRACTION
    intent_slo_pass = results.intent_accuracy >= SLO_ROUTER_ACCURACY
    ctx_precision_slo_pass = results.avg_context_precision >= SLO_FLOW_CONTEXT_PRECISION
    relevance_slo_pass = results.avg_relevance >= SLO_FLOW_RELEVANCE
    faithfulness_slo_pass = results.avg_faithfulness >= SLO_FLOW_FAITHFULNESS
    answer_correctness_slo_pass = results.avg_answer_correctness >= SLO_FLOW_ANSWER_CORRECTNESS

    # Compute latency SLO pass/fail
    routing_latency_pass = results.latency_routing_pct <= SLO_LATENCY_ROUTING_PCT
    retrieval_latency_pass = results.latency_retrieval_pct <= SLO_LATENCY_RETRIEVAL_PCT
    answer_latency_pass = results.latency_answer_pct <= SLO_LATENCY_ANSWER_PCT

    # Build table sections matching E2E structure: (section_name, [(label, value, slo_target, slo_passed)])
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]] = [
        (
            "Routing",
            [
                (
                    "  Company Extraction",
                    format_percentage(results.company_extraction_accuracy),
                    f">={format_percentage(SLO_COMPANY_EXTRACTION)}",
                    company_slo_pass,
                ),
                (
                    "  Intent Classification",
                    format_percentage(results.intent_accuracy),
                    f">={format_percentage(SLO_ROUTER_ACCURACY)}",
                    intent_slo_pass,
                ),
                (
                    "  latency",
                    format_percentage(results.latency_routing_pct),
                    f"<={format_percentage(SLO_LATENCY_ROUTING_PCT)}",
                    routing_latency_pass,
                ),
            ],
        ),
        (
            "Retrieval",
            [
                (
                    "  Context Precision",
                    format_percentage(results.avg_context_precision),
                    f">={format_percentage(SLO_FLOW_CONTEXT_PRECISION)}",
                    ctx_precision_slo_pass,
                ),
                (
                    "  latency",
                    format_percentage(results.latency_retrieval_pct),
                    f"<={format_percentage(SLO_LATENCY_RETRIEVAL_PCT)}",
                    retrieval_latency_pass,
                ),
            ],
        ),
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
                (
                    "  latency",
                    format_percentage(results.latency_answer_pct),
                    f"<={format_percentage(SLO_LATENCY_ANSWER_PCT)}",
                    answer_latency_pass,
                ),
            ],
        ),
    ]

    summary_table = build_eval_table("Flow Evaluation Summary", sections)
    console.print(summary_table)

    # Latency stats below table
    console.print(
        f"\nLatency: {results.wall_clock_ms / 1000:.1f}s total | "
        f"{results.avg_latency_per_question_ms:.0f}ms avg | "
        f"{results.p95_latency_ms:.0f}ms P95"
    )

    all_slos_passed = (
        path_slo_pass
        and q_slo_pass
        and company_slo_pass
        and intent_slo_pass
        and ctx_precision_slo_pass
        and relevance_slo_pass
        and faithfulness_slo_pass
        and answer_correctness_slo_pass
        and routing_latency_pass
        and retrieval_latency_pass
        and answer_latency_pass
    )

    # SLO Failures Detail
    _print_slo_failures(results)

    return all_slos_passed


def _count_ragas_failures(step: FlowStepResult) -> int:
    """Count how many RAGAS metrics failed for a step."""
    count = 0
    if step.relevance_score < SLO_FLOW_RELEVANCE:
        count += 1
    if step.faithfulness_score < SLO_FLOW_FAITHFULNESS:
        count += 1
    if step.context_precision_score < SLO_FLOW_CONTEXT_PRECISION:
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
            if _count_ragas_failures(step) > 0:
                failures.append((flow_result.path_id, step))

    if not failures:
        return

    # Sort by failure count (most failures first)
    failures.sort(key=lambda x: _count_ragas_failures(x[1]), reverse=True)

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
    failed_table.add_column("Question", width=45)
    failed_table.add_column("R", justify="center", width=3)
    failed_table.add_column("F", justify="center", width=3)
    failed_table.add_column("C", justify="center", width=3)
    failed_table.add_column("A", justify="center", width=3)

    def fmt(passed: bool) -> str:
        return "[green]Y[/green]" if passed else "[red]X[/red]"

    for path_id, step in shown:
        question_display = step.question[:43] + "..." if len(step.question) > 43 else step.question

        r_pass = step.relevance_score >= SLO_FLOW_RELEVANCE
        f_pass = step.faithfulness_score >= SLO_FLOW_FAITHFULNESS
        c_pass = step.context_precision_score >= SLO_FLOW_CONTEXT_PRECISION
        a_pass = step.answer_correctness_score >= SLO_FLOW_ANSWER_CORRECTNESS

        failed_table.add_row(
            str(path_id + 1),
            question_display,
            fmt(r_pass),
            fmt(f_pass),
            fmt(c_pass),
            fmt(a_pass),
        )

    console.print(failed_table)


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
            "company_extraction_accuracy": results.company_extraction_accuracy,
            "intent_accuracy": results.intent_accuracy,
            "avg_relevance": results.avg_relevance,
            "avg_faithfulness": results.avg_faithfulness,
            "avg_context_precision": results.avg_context_precision,
            "avg_answer_correctness": results.avg_answer_correctness,
            "total_latency_ms": results.total_latency_ms,
            "avg_latency_per_question_ms": results.avg_latency_per_question_ms,
            "p95_latency_ms": results.p95_latency_ms,
            "wall_clock_ms": results.wall_clock_ms,
            "latency_routing_pct": results.latency_routing_pct,
            "latency_retrieval_pct": results.latency_retrieval_pct,
            "latency_answer_pct": results.latency_answer_pct,
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
            "context_precision": {
                "value": results.avg_context_precision,
                "target": SLO_FLOW_CONTEXT_PRECISION,
                "passed": results.avg_context_precision >= SLO_FLOW_CONTEXT_PRECISION,
            },
            "answer_correctness": {
                "value": results.avg_answer_correctness,
                "target": SLO_FLOW_ANSWER_CORRECTNESS,
                "passed": results.avg_answer_correctness >= SLO_FLOW_ANSWER_CORRECTNESS,
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
                        "faithfulness_score": s.faithfulness_score,
                        "context_precision_score": s.context_precision_score,
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
        from backend.agent.rag.client import get_qdrant_client

        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        if "already accessed" in str(e).lower():
            return False
        logger.warning(f"Qdrant check error: {e}")
        return True
