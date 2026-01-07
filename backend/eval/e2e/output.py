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
    SLO_SECURITY_PASS_RATE,
    E2EEvalResult,
    E2EEvalSummary,
)


def print_e2e_eval_results(
    results: list[E2EEvalResult],
    summary: E2EEvalSummary,
) -> None:
    """Print end-to-end evaluation results."""
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

    # Security Tests and Latency below table
    security_slo_pass = summary.security_pass_rate >= SLO_SECURITY_PASS_RATE
    security_color = "green" if security_slo_pass else "red"
    console.print(
        f"\nSecurity Tests: [{security_color}]{summary.security_tests_passed}/{summary.security_tests_total}[/{security_color}] passed"
    )
    console.print(
        f"Latency: {summary.wall_clock_ms / 1000:.1f}s total | {summary.avg_latency_ms:.0f}ms avg/question"
    )

    # By-category breakdown
    console.print()
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

    # Issue Details
    _print_issues(results)


def _get_failed_slos(r: E2EEvalResult) -> list[str]:
    """Get list of failed SLOs for a result."""
    failures = []

    # Routing SLOs (only check if expected value exists)
    if r.expected_company_id and not r.company_correct:
        failures.append(f"Company: expected={r.expected_company_id}, got={r.actual_company_id}")
    if r.expected_intent and not r.intent_correct:
        failures.append(f"Intent: expected={r.expected_intent}, got={r.actual_intent}")

    # RAGAS SLOs (skip for security tests)
    if r.category not in SECURITY_CATEGORIES:
        if r.answer_relevance < SLO_ANSWER_RELEVANCE:
            failures.append(f"Relevance: {r.answer_relevance:.0%} < {SLO_ANSWER_RELEVANCE:.0%}")
        if r.faithfulness < SLO_FAITHFULNESS:
            failures.append(f"Faithfulness: {r.faithfulness:.0%} < {SLO_FAITHFULNESS:.0%}")
        if r.context_precision < SLO_CONTEXT_PRECISION:
            failures.append(f"CtxPrecision: {r.context_precision:.0%} < {SLO_CONTEXT_PRECISION:.0%}")
        if r.answer_correctness < SLO_ANSWER_CORRECTNESS:
            failures.append(f"AnsCorrectness: {r.answer_correctness:.0%} < {SLO_ANSWER_CORRECTNESS:.0%}")

    return failures


def _print_issues(results: list[E2EEvalResult]) -> None:
    """Print details of SLO failures."""
    issues = []
    for i, r in enumerate(results, 1):
        failures = _get_failed_slos(r)
        if failures:
            issues.append((i, r, failures))

    if not issues:
        return

    console.print(f"\n[yellow bold]SLO Failures ({len(issues)} questions):[/yellow bold]")
    for i, r, failures in issues:
        # Show test_case_id if question is empty or too short
        if len(r.question.strip()) > 10:
            question_display = r.question[:60] + ("..." if len(r.question) > 60 else "")
        else:
            question_display = f"\\[{r.test_case_id}]"
        console.print(f"\n  [#{i}] {question_display}")
        for f in failures:
            console.print(f"    [red]x[/red] {f}")
