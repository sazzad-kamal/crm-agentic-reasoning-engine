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
    SLO_SQL_DATA,
    SLO_SQL_SUCCESS,
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
    # SQL generation success: N/A if no queries were executed
    sql_slo_pass = results.sql_success_rate >= SLO_SQL_SUCCESS if results.sql_query_count > 0 else None
    # SQL data validation: N/A if no assertions defined
    sql_data_slo_pass = results.sql_data_success_rate >= SLO_SQL_DATA if results.sql_data_count > 0 else None

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

    # RAG detection SLO (90% accuracy)
    rag_detection_slo_pass = results.rag_detection_rate >= 0.90 if results.rag_detection_count > 0 else None

    # Build table sections matching E2E structure: (section_name, [(label, value, slo_target, slo_passed)])
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]] = [
        (
            "Routing",
            [
                (
                    "  SQL Generation",
                    format_percentage(results.sql_success_rate) if results.sql_query_count > 0 else "N/A",
                    f">={format_percentage(SLO_SQL_SUCCESS)}" if results.sql_query_count > 0 else "-",
                    sql_slo_pass,
                ),
                (
                    "  SQL Data",
                    format_percentage(results.sql_data_success_rate) if results.sql_data_count > 0 else "N/A",
                    f">={format_percentage(SLO_SQL_DATA)}" if results.sql_data_count > 0 else "-",
                    sql_data_slo_pass,
                ),
                (
                    "  RAG Detection",
                    format_percentage(results.rag_detection_rate) if results.rag_detection_count > 0 else "N/A",
                    ">=90.0%" if results.rag_detection_count > 0 else "-",
                    rag_detection_slo_pass,
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
        and (sql_slo_pass is None or sql_slo_pass)
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


def _count_slo_failures(step: FlowStepResult, eval_mode: str = "both") -> int:
    """Count how many SLO metrics failed for a step based on eval_mode."""
    count = 0
    # SQL execution success
    if step.sql_queries_total > 0 and step.sql_queries_success < step.sql_queries_total:
        count += 1
    # SQL data validation (only count if assertions exist and failed)
    if step.sql_data_validated is False:
        count += 1
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
            if _count_slo_failures(step, eval_mode) > 0:
                failures.append((flow_result.path_id, step))

    if not failures:
        return

    # Sort by failure count (most failures first)
    failures.sort(key=lambda x: _count_slo_failures(x[1], eval_mode), reverse=True)

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
    failed_table.add_column("Question", width=32)
    # Always show SQL Gen and SQL Data columns
    failed_table.add_column("SQL Gen", justify="center", width=7)
    failed_table.add_column("SQL Data", justify="center", width=8)

    # Add columns based on eval_mode
    if eval_mode in ("pipeline", "both"):
        failed_table.add_column("R", justify="center", width=3)
        failed_table.add_column("F", justify="center", width=3)
        failed_table.add_column("A", justify="center", width=3)
    if eval_mode in ("rag", "both"):
        failed_table.add_column("Precision", justify="center", width=9)
        failed_table.add_column("Recall", justify="center", width=6)

    def fmt(passed: bool | None) -> str:
        if passed is None:
            return "[dim]-[/dim]"
        return "[green]Y[/green]" if passed else "[red]X[/red]"

    for path_id, step in shown:
        question_display = step.question[:30] + "..." if len(step.question) > 30 else step.question

        # SQL execution success: Y if no queries or all succeeded, X if any failed
        sql_gen_pass = step.sql_queries_total == 0 or step.sql_queries_success == step.sql_queries_total
        # SQL data validation: Y if passed, X if failed, - if no assertions defined
        sql_data_pass = step.sql_data_validated  # None = no assertions, True = passed, False = failed

        row: list[str] = [str(path_id + 1), question_display, fmt(sql_gen_pass), fmt(sql_data_pass)]

        if eval_mode in ("pipeline", "both"):
            r_pass = step.relevance_score >= SLO_FLOW_RELEVANCE
            f_pass = step.faithfulness_score >= SLO_FLOW_FAITHFULNESS
            a_pass = step.answer_correctness_score >= SLO_FLOW_ANSWER_CORRECTNESS
            row.extend([fmt(r_pass), fmt(f_pass), fmt(a_pass)])

        if eval_mode in ("rag", "both"):
            # Precision/Recall: N/A if not invoked, Y if passed, X if failed
            p_pass: bool | None = None if not step.account_rag_invoked else (
                step.account_precision_score >= SLO_ACCOUNT_PRECISION
            )
            rc_pass: bool | None = None if not step.account_rag_invoked else (
                step.account_recall_score >= SLO_ACCOUNT_RECALL
            )
            row.extend([fmt(p_pass), fmt(rc_pass)])

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
            "sql_success_rate": results.sql_success_rate,
            "sql_query_count": results.sql_query_count,
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
                        "sql_queries_total": s.sql_queries_total,
                        "sql_queries_success": s.sql_queries_success,
                        "sql_data_validated": s.sql_data_validated,
                        "sql_data_errors": s.sql_data_errors,
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
