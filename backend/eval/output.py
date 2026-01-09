"""Output and display functions for evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.table import Table

from backend.eval.formatting import build_eval_table, console, format_percentage
from backend.eval.models import (
    SLO_ACCOUNT_PRECISION,
    SLO_ACCOUNT_RECALL,
    SLO_COMPANY_EXTRACTION,
    SLO_FLOW_ANSWER_CORRECTNESS,
    SLO_FLOW_AVG_LATENCY_MS,
    SLO_FLOW_COMPOSITE_SCORE,
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


def print_summary(results: FlowEvalResults, eval_mode: str = "both") -> bool:
    """
    Print a comprehensive summary of eval results with SLO status.

    Args:
        results: The evaluation results
        eval_mode: RAGAS mode - 'rag', 'pipeline', or 'both'

    Returns:
        True if all SLOs passed
    """
    console.print()

    # Compute SLO pass/fail
    path_slo_pass = results.path_pass_rate >= SLO_FLOW_PATH_PASS_RATE
    q_slo_pass = results.question_pass_rate >= SLO_FLOW_QUESTION_PASS_RATE
    # Company extraction: N/A if no questions have expected company
    company_slo_pass = results.company_extraction_accuracy >= SLO_COMPANY_EXTRACTION if results.company_sample_count > 0 else None
    intent_slo_pass = results.intent_accuracy >= SLO_ROUTER_ACCURACY

    # Answer quality metrics - N/A if eval_mode is "rag" (not evaluated)
    if eval_mode == "rag":
        relevance_slo_pass = None
        faithfulness_slo_pass = None
        answer_correctness_slo_pass = None
    else:
        relevance_slo_pass = results.avg_relevance >= SLO_FLOW_RELEVANCE
        faithfulness_slo_pass = results.avg_faithfulness >= SLO_FLOW_FAITHFULNESS
        answer_correctness_slo_pass = results.avg_answer_correctness >= SLO_FLOW_ANSWER_CORRECTNESS

    # Account RAG SLOs - N/A if eval_mode is "pipeline" or never invoked
    if eval_mode == "pipeline":
        account_precision_slo_pass = None
        account_recall_slo_pass = None
    else:
        account_precision_slo_pass = results.avg_account_precision >= SLO_ACCOUNT_PRECISION if results.account_sample_count > 0 else None
        account_recall_slo_pass = results.avg_account_recall >= SLO_ACCOUNT_RECALL if results.account_sample_count > 0 else None

    # Compute latency SLO pass/fail
    routing_latency_pass = results.latency_routing_pct <= SLO_LATENCY_ROUTING_PCT
    retrieval_latency_pass = results.latency_retrieval_pct <= SLO_LATENCY_RETRIEVAL_PCT
    answer_latency_pass = results.latency_answer_pct <= SLO_LATENCY_ANSWER_PCT
    avg_latency_pass = results.avg_latency_per_question_ms <= SLO_FLOW_AVG_LATENCY_MS

    # Composite score pass/fail - only valid when all metrics are evaluated
    if eval_mode == "both":
        composite_pass = results.composite_score >= SLO_FLOW_COMPOSITE_SCORE
    else:
        composite_pass = None  # N/A when not all metrics evaluated

    # Build table sections matching E2E structure: (section_name, [(label, value, slo_target, slo_passed)])
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]] = [
        (
            "Routing",
            [
                (
                    "  Company Extraction",
                    format_percentage(results.company_extraction_accuracy) if results.company_sample_count > 0 else "N/A",
                    f">={format_percentage(SLO_COMPANY_EXTRACTION)}" if results.company_sample_count > 0 else "-",
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
            f"Fetch ({results.account_sample_count}/{results.rag_invoked_count})" if results.rag_invoked_count > 0 and eval_mode != "pipeline" else "Fetch",
            [
                (
                    "  Precision",
                    format_percentage(results.avg_account_precision) if results.account_sample_count > 0 and eval_mode != "pipeline" else "N/A",
                    f">={format_percentage(SLO_ACCOUNT_PRECISION)}" if results.account_sample_count > 0 and eval_mode != "pipeline" else "-",
                    account_precision_slo_pass,
                ),
                (
                    "  Recall",
                    format_percentage(results.avg_account_recall) if results.account_sample_count > 0 and eval_mode != "pipeline" else "N/A",
                    f">={format_percentage(SLO_ACCOUNT_RECALL)}" if results.account_sample_count > 0 and eval_mode != "pipeline" else "-",
                    account_recall_slo_pass,
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
                    format_percentage(results.avg_relevance) if eval_mode != "rag" else "N/A",
                    f">={format_percentage(SLO_FLOW_RELEVANCE)}" if eval_mode != "rag" else "-",
                    relevance_slo_pass,
                ),
                (
                    "  Faithfulness",
                    format_percentage(results.avg_faithfulness) if eval_mode != "rag" else "N/A",
                    f">={format_percentage(SLO_FLOW_FAITHFULNESS)}" if eval_mode != "rag" else "-",
                    faithfulness_slo_pass,
                ),
                (
                    "  Answer Correctness",
                    format_percentage(results.avg_answer_correctness) if eval_mode != "rag" else "N/A",
                    f">={format_percentage(SLO_FLOW_ANSWER_CORRECTNESS)}" if eval_mode != "rag" else "-",
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

    # Build table with composite score as aggregate row
    summary_table = build_eval_table(
        "Flow Evaluation Summary",
        sections,
        aggregate_row=(
            "Composite Score",
            format_percentage(results.composite_score) if eval_mode == "both" else "N/A",
            f">={format_percentage(SLO_FLOW_COMPOSITE_SCORE)}" if eval_mode == "both" else "-",
            composite_pass,
        ),
    )
    console.print(summary_table)

    # N/A metrics (None) don't affect overall pass/fail
    all_slos_passed = (
        path_slo_pass
        and q_slo_pass
        and (company_slo_pass is None or company_slo_pass)
        and intent_slo_pass
        and (account_precision_slo_pass is None or account_precision_slo_pass)
        and (account_recall_slo_pass is None or account_recall_slo_pass)
        and relevance_slo_pass
        and faithfulness_slo_pass
        and answer_correctness_slo_pass
        and routing_latency_pass
        and retrieval_latency_pass
        and answer_latency_pass
        and avg_latency_pass
        and composite_pass
    )

    # SLO Failures Detail
    _print_slo_failures(results, eval_mode)

    return all_slos_passed


def _count_ragas_failures(step: FlowStepResult, eval_mode: str = "both") -> int:
    """Count how many RAGAS metrics failed for a step based on eval_mode."""
    count = 0
    # Pipeline metrics (relevance, faithfulness, answer_correctness)
    if eval_mode in ("pipeline", "both"):
        if step.relevance_score < SLO_FLOW_RELEVANCE:
            count += 1
        if step.faithfulness_score < SLO_FLOW_FAITHFULNESS:
            count += 1
        if step.answer_correctness_score < SLO_FLOW_ANSWER_CORRECTNESS:
            count += 1
    # RAG metrics (precision, recall) - only count if invoked
    if eval_mode in ("rag", "both"):
        if step.account_rag_invoked and step.account_precision_score < SLO_ACCOUNT_PRECISION:
            count += 1
        if step.account_rag_invoked and step.account_recall_score < SLO_ACCOUNT_RECALL:
            count += 1
    return count


def _print_slo_failures(results: FlowEvalResults, eval_mode: str = "both") -> None:
    """Print details of SLO failures as a compact table."""
    # Collect all failed steps with their path info
    failures: list[tuple[int, FlowStepResult]] = []
    for flow_result in results.all_results:
        for step in flow_result.steps:
            if _count_ragas_failures(step, eval_mode) > 0:
                failures.append((flow_result.path_id, step))

    if not failures:
        return

    # Sort by failure count (most failures first)
    failures.sort(key=lambda x: _count_ragas_failures(x[1], eval_mode), reverse=True)

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

    # Add columns based on eval_mode
    if eval_mode in ("pipeline", "both"):
        failed_table.add_column("R", justify="center", width=3)
        failed_table.add_column("F", justify="center", width=3)
        failed_table.add_column("A", justify="center", width=3)
    if eval_mode in ("rag", "both"):
        failed_table.add_column("P/R", justify="center", width=4)

    def fmt(passed: bool | None) -> str:
        if passed is None:
            return "[dim]-[/dim]"
        return "[green]Y[/green]" if passed else "[red]X[/red]"

    for path_id, step in shown:
        question_display = step.question[:38] + "..." if len(step.question) > 38 else step.question

        row: list[str] = [str(path_id + 1), question_display]

        if eval_mode in ("pipeline", "both"):
            r_pass = step.relevance_score >= SLO_FLOW_RELEVANCE
            f_pass = step.faithfulness_score >= SLO_FLOW_FAITHFULNESS
            a_pass = step.answer_correctness_score >= SLO_FLOW_ANSWER_CORRECTNESS
            row.extend([fmt(r_pass), fmt(f_pass), fmt(a_pass)])

        if eval_mode in ("rag", "both"):
            # Account: N/A if not invoked, else check both precision and recall
            acct_pass: bool | None = None if not step.account_rag_invoked else (
                step.account_precision_score >= SLO_ACCOUNT_PRECISION and step.account_recall_score >= SLO_ACCOUNT_RECALL
            )
            row.append(fmt(acct_pass))

        failed_table.add_row(*row)

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
            "company_extraction_accuracy": results.company_extraction_accuracy,
            "intent_accuracy": results.intent_accuracy,
            "avg_relevance": results.avg_relevance,
            "avg_faithfulness": results.avg_faithfulness,
            "avg_answer_correctness": results.avg_answer_correctness,
            "avg_precision": results.avg_account_precision,
            "avg_recall": results.avg_account_recall,
            "total_latency_ms": results.total_latency_ms,
            "avg_latency_per_question_ms": results.avg_latency_per_question_ms,
            "wall_clock_ms": results.wall_clock_ms,
            "latency_routing_pct": results.latency_routing_pct,
            "latency_retrieval_pct": results.latency_retrieval_pct,
            "latency_answer_pct": results.latency_answer_pct,
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
            "precision": {
                "value": results.avg_account_precision,
                "target": SLO_ACCOUNT_PRECISION,
                "passed": results.avg_account_precision >= SLO_ACCOUNT_PRECISION,
            },
            "recall": {
                "value": results.avg_account_recall,
                "target": SLO_ACCOUNT_RECALL,
                "passed": results.avg_account_recall >= SLO_ACCOUNT_RECALL,
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
                        "precision_score": s.account_precision_score,
                        "recall_score": s.account_recall_score,
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
