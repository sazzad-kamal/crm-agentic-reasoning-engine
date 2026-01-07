"""Output and display functions for E2E evaluation."""

from __future__ import annotations

from rich.table import Table

from backend.eval.base import console, format_percentage
from backend.eval.formatting import build_eval_table
from backend.eval.models import (
    SECURITY_CATEGORIES,
    SLO_ANSWER_CORRECTNESS,
    SLO_ANSWER_RELEVANCE,
    SLO_COMPANY_EXTRACTION,
    SLO_CONTEXT_PRECISION,
    SLO_FAITHFULNESS,
    SLO_LATENCY_ANSWER_PCT,
    SLO_LATENCY_FOLLOWUP_PCT,
    SLO_LATENCY_RETRIEVAL_PCT,
    SLO_LATENCY_ROUTING_PCT,
    SLO_ROUTER_ACCURACY,
    E2EEvalResult,
    E2EEvalSummary,
)


def print_e2e_eval_results(
    results: list[E2EEvalResult],
    summary: E2EEvalSummary,
) -> None:
    """Print end-to-end evaluation results."""
    # 1. Category table first
    cat_table = Table(title="Results by Category", show_header=True)
    cat_table.add_column("Category")
    cat_table.add_column("Tests", justify="right")
    cat_table.add_column("Passed", justify="right")
    cat_table.add_column("Pass%", justify="right")

    for cat, stats in sorted(summary.by_category.items()):
        pass_rate = stats["passed"] / stats["count"] if stats["count"] > 0 else 0
        cat_table.add_row(
            cat,
            str(stats["count"]),
            str(stats["passed"]),
            format_percentage(pass_rate),
        )

    console.print(cat_table)
    console.print()

    # 2. Main summary table
    # Compute SLO pass/fail for quality metrics
    company_slo_pass = summary.company_extraction_accuracy >= SLO_COMPANY_EXTRACTION
    intent_slo_pass = summary.intent_accuracy >= SLO_ROUTER_ACCURACY
    ctx_precision_slo_pass = summary.context_precision_rate >= SLO_CONTEXT_PRECISION
    relevance_slo_pass = summary.answer_relevance_rate >= SLO_ANSWER_RELEVANCE
    faithfulness_slo_pass = summary.faithfulness_rate >= SLO_FAITHFULNESS
    answer_correctness_slo_pass = summary.answer_correctness_rate >= SLO_ANSWER_CORRECTNESS

    # Compute SLO pass/fail for latency percentages
    routing_latency_pass = summary.latency_routing_pct <= SLO_LATENCY_ROUTING_PCT
    retrieval_latency_pass = summary.latency_retrieval_pct <= SLO_LATENCY_RETRIEVAL_PCT
    answer_latency_pass = summary.latency_answer_pct <= SLO_LATENCY_ANSWER_PCT
    followup_latency_pass = summary.latency_followup_pct <= SLO_LATENCY_FOLLOWUP_PCT

    # Build table sections: (section_name, [(label, value, slo_target, slo_passed)])
    sections: list[tuple[str, list[tuple[str, str, str | None, bool | None]]]] = [
        (
            "Routing",
            [
                (
                    "  Company Extraction",
                    format_percentage(summary.company_extraction_accuracy),
                    f">={format_percentage(SLO_COMPANY_EXTRACTION)}",
                    company_slo_pass,
                ),
                (
                    "  Intent Classification",
                    format_percentage(summary.intent_accuracy),
                    f">={format_percentage(SLO_ROUTER_ACCURACY)}",
                    intent_slo_pass,
                ),
                (
                    "  latency",
                    format_percentage(summary.latency_routing_pct),
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
                    format_percentage(summary.context_precision_rate),
                    f">={format_percentage(SLO_CONTEXT_PRECISION)}",
                    ctx_precision_slo_pass,
                ),
                (
                    "  latency",
                    format_percentage(summary.latency_retrieval_pct),
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
                    format_percentage(summary.answer_relevance_rate),
                    f">={format_percentage(SLO_ANSWER_RELEVANCE)}",
                    relevance_slo_pass,
                ),
                (
                    "  Faithfulness",
                    format_percentage(summary.faithfulness_rate),
                    f">={format_percentage(SLO_FAITHFULNESS)}",
                    faithfulness_slo_pass,
                ),
                (
                    "  Answer Correctness",
                    format_percentage(summary.answer_correctness_rate),
                    f">={format_percentage(SLO_ANSWER_CORRECTNESS)}",
                    answer_correctness_slo_pass,
                ),
                (
                    "  latency",
                    format_percentage(summary.latency_answer_pct),
                    f"<={format_percentage(SLO_LATENCY_ANSWER_PCT)}",
                    answer_latency_pass,
                ),
            ],
        ),
        (
            "Followup",
            [
                (
                    "  latency",
                    format_percentage(summary.latency_followup_pct),
                    f"<={format_percentage(SLO_LATENCY_FOLLOWUP_PCT)}",
                    followup_latency_pass,
                ),
            ],
        ),
    ]

    table = build_eval_table("E2E Evaluation Summary", sections)
    console.print(table)

    # 3. Latency line
    console.print(
        f"Latency: {summary.wall_clock_ms / 1000:.1f}s total | {summary.avg_latency_ms:.0f}ms avg/question"
    )

    # 4. SLO Failures table
    _print_issues(results)


def _count_ragas_failures(r: E2EEvalResult) -> int:
    """Count how many RAGAS metrics failed for a result."""
    if r.category in SECURITY_CATEGORIES:
        return 0
    count = 0
    if r.answer_relevance < SLO_ANSWER_RELEVANCE:
        count += 1
    if r.faithfulness < SLO_FAITHFULNESS:
        count += 1
    if r.context_precision < SLO_CONTEXT_PRECISION:
        count += 1
    if r.answer_correctness < SLO_ANSWER_CORRECTNESS:
        count += 1
    return count


def _print_issues(results: list[E2EEvalResult]) -> None:
    """Print details of SLO failures as a compact table."""
    # Collect failures (non-security tests with at least one RAGAS failure)
    failures = [
        r for r in results
        if r.category not in SECURITY_CATEGORIES and _count_ragas_failures(r) > 0
    ]

    if not failures:
        return

    # Sort by failure count (most failures first)
    failures.sort(key=lambda r: _count_ragas_failures(r), reverse=True)

    # Show top 5
    shown = failures[:5]
    total = len(failures)

    console.print()
    failed_table = Table(
        title=f"SLO Failures ({len(shown)} of {total} shown, sorted by severity)",
        show_header=True,
        header_style="bold yellow",
    )
    failed_table.add_column("#", style="dim", width=3)
    failed_table.add_column("Question", width=45)
    failed_table.add_column("R", justify="center", width=3)
    failed_table.add_column("F", justify="center", width=3)
    failed_table.add_column("C", justify="center", width=3)
    failed_table.add_column("A", justify="center", width=3)

    for i, r in enumerate(shown, 1):
        # Format question display (show question, fallback to test_case_id if empty)
        if r.question.strip():
            question_display = r.question[:43] + "..." if len(r.question) > 43 else r.question
        else:
            question_display = f"[{r.test_case_id}]"

        # Format RAGAS metrics as checkmarks
        r_pass = r.answer_relevance >= SLO_ANSWER_RELEVANCE
        f_pass = r.faithfulness >= SLO_FAITHFULNESS
        c_pass = r.context_precision >= SLO_CONTEXT_PRECISION
        a_pass = r.answer_correctness >= SLO_ANSWER_CORRECTNESS

        def fmt(passed: bool) -> str:
            return "[green]Y[/green]" if passed else "[red]X[/red]"

        failed_table.add_row(
            str(i),
            question_display,
            fmt(r_pass),
            fmt(f_pass),
            fmt(c_pass),
            fmt(a_pass),
        )

    console.print(failed_table)
