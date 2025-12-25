"""
End-to-end agent evaluation harness.

Tests the full orchestrator pipeline:
- Question → Router → Tools → RAG → LLM → Answer
- Evaluates answer quality using LLM-as-judge
- Tracks tool selection, latency, and cost

Usage:
    python -m backend.agent.eval.e2e_eval
    python -m backend.agent.eval.e2e_eval --verbose
    python -m backend.agent.eval.e2e_eval --limit 10
"""

import json
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

from backend.agent.orchestrator import answer_question
from backend.agent.eval.models import E2EEvalResult, E2EEvalSummary
from backend.common.llm_client import call_llm


console = Console()


# =============================================================================
# LLM Judge for E2E
# =============================================================================

E2E_JUDGE_SYSTEM = """You are an expert evaluator for a CRM assistant.
Evaluate the quality of the assistant's answer to a user question.

Score each criterion as 0 or 1:

1. ANSWER_RELEVANCE: Does the answer address the user's question?
   - 1 if the answer directly addresses what was asked
   - 0 if the answer is off-topic or doesn't address the question

2. ANSWER_GROUNDED: Is the answer based on the provided sources/data?
   - 1 if the answer appears grounded in real data (mentions specific companies, dates, values)
   - 0 if the answer seems made up or generic

Respond in JSON:
{
  "answer_relevance": 0 or 1,
  "answer_grounded": 0 or 1,
  "explanation": "brief explanation"
}"""

E2E_JUDGE_PROMPT = """Question: {question}

Answer: {answer}

Sources cited: {sources}

Evaluate this response:"""


def judge_e2e_response(
    question: str,
    answer: str,
    sources: list[str],
) -> dict:
    """Judge an end-to-end response using LLM."""
    prompt = E2E_JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        sources=", ".join(sources) if sources else "None",
    )
    
    try:
        response = call_llm(
            prompt,
            system_prompt=E2E_JUDGE_SYSTEM,
            model="o4-mini",  # Use reasoning model for evaluation
            temperature=0.0,
            max_tokens=200,
        )
        
        # Parse JSON from response (call_llm returns a string)
        text = response
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        result = json.loads(text.strip())
        return {
            "answer_relevance": result.get("answer_relevance", 0),
            "answer_grounded": result.get("answer_grounded", 0),
            "explanation": result.get("explanation", ""),
        }
    except Exception as e:
        return {
            "answer_relevance": 0,
            "answer_grounded": 0,
            "explanation": f"Judge error: {str(e)}",
        }


# =============================================================================
# Test Cases
# =============================================================================

E2E_TEST_CASES = [
    # Data-focused questions
    {
        "id": "e2e_data_status",
        "question": "What's the current status of Acme Manufacturing?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["company_lookup"],
    },
    {
        "id": "e2e_data_activity",
        "question": "Show me recent activities for Beta Tech Solutions",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["company_lookup", "recent_activity"],
    },
    {
        "id": "e2e_data_history",
        "question": "What calls and emails have we had with Crown Foods?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["company_lookup", "recent_history"],
    },
    {
        "id": "e2e_data_pipeline",
        "question": "What opportunities are in the pipeline for Delta Health?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["company_lookup", "pipeline"],
    },
    {
        "id": "e2e_data_renewals",
        "question": "What renewals are coming up in the next 90 days?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["upcoming_renewals"],
    },
    {
        "id": "e2e_data_churned",
        "question": "What happened with Green Energy Partners? Why did they churn?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["company_lookup", "recent_history"],
    },
    # Docs-focused questions
    {
        "id": "e2e_docs_howto",
        "question": "How do I create a new contact in Acme CRM?",
        "category": "docs",
        "expected_mode": "docs",
        "expected_tools": [],
    },
    {
        "id": "e2e_docs_explain",
        "question": "What are the different pipeline stages in Acme CRM?",
        "category": "docs",
        "expected_mode": "docs",
        "expected_tools": [],
    },
    {
        "id": "e2e_docs_feature",
        "question": "How does the email marketing campaign feature work?",
        "category": "docs",
        "expected_mode": "docs",
        "expected_tools": [],
    },
    # Combined questions
    {
        "id": "e2e_combined_1",
        "question": "How do I track renewal risk, and what's the renewal status for Acme Manufacturing?",
        "category": "combined",
        "expected_mode": "data+docs",
        "expected_tools": ["company_lookup"],
    },
    {
        "id": "e2e_combined_2",
        "question": "What pipeline stages is Fusion Retail in, and how do I move deals between stages?",
        "category": "combined",
        "expected_mode": "data+docs",
        "expected_tools": ["company_lookup", "pipeline"],
    },
    # Complex questions
    {
        "id": "e2e_complex_summary",
        "question": "Give me a complete summary of Harbor Logistics - their status, contacts, activities, and opportunities",
        "category": "complex",
        "expected_mode": "data",
        "expected_tools": ["company_lookup", "recent_activity", "recent_history", "pipeline"],
    },
    {
        "id": "e2e_complex_risk",
        "question": "Which accounts are at risk of churning and what should I do about them?",
        "category": "complex",
        "expected_mode": "data+docs",
        "expected_tools": ["upcoming_renewals"],
    },
    # Edge cases
    {
        "id": "e2e_edge_partial",
        "question": "What's going on with Eastern?",
        "category": "edge",
        "expected_mode": "data",
        "expected_tools": ["company_lookup"],
    },
    {
        "id": "e2e_edge_ambiguous",
        "question": "Tell me about opportunities",
        "category": "edge",
        "expected_mode": "data+docs",
        "expected_tools": [],
    },
    # =========================================================================
    # NEW TOOL COVERAGE - Contact Search
    # =========================================================================
    {
        "id": "e2e_contacts_decision_makers",
        "question": "Who are the decision makers across all our accounts?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_contacts"],
    },
    {
        "id": "e2e_contacts_company",
        "question": "Show me the contacts at Beta Tech Solutions",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_contacts"],
    },
    {
        "id": "e2e_contacts_champions",
        "question": "List all champion contacts",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_contacts"],
    },
    # =========================================================================
    # NEW TOOL COVERAGE - Company Search
    # =========================================================================
    {
        "id": "e2e_companies_enterprise",
        "question": "Show me all enterprise accounts",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_companies"],
    },
    {
        "id": "e2e_companies_industry",
        "question": "Which companies are in the manufacturing industry?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_companies"],
    },
    {
        "id": "e2e_companies_smb",
        "question": "List all SMB companies",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_companies"],
    },
    # =========================================================================
    # NEW TOOL COVERAGE - Groups
    # =========================================================================
    {
        "id": "e2e_groups_at_risk",
        "question": "Who is in the at-risk accounts group?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["group_members"],
    },
    {
        "id": "e2e_groups_list",
        "question": "What groups do we have?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["list_groups"],
    },
    {
        "id": "e2e_groups_churned",
        "question": "Show the churned accounts group",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["group_members"],
    },
    # =========================================================================
    # NEW TOOL COVERAGE - Pipeline Summary (Aggregate)
    # =========================================================================
    {
        "id": "e2e_pipeline_total",
        "question": "What's the total pipeline value?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["pipeline_summary"],
    },
    {
        "id": "e2e_pipeline_deals_count",
        "question": "How many deals do we have in the pipeline?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["pipeline_summary"],
    },
    {
        "id": "e2e_pipeline_forecast",
        "question": "Give me a pipeline overview across all accounts",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["pipeline_summary"],
    },
    # =========================================================================
    # NEW TOOL COVERAGE - Attachments
    # =========================================================================
    {
        "id": "e2e_attachments_proposals",
        "question": "Find all proposals",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_attachments"],
    },
    {
        "id": "e2e_attachments_company",
        "question": "What documents do we have for Crown Foods?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_attachments"],
    },
    {
        "id": "e2e_attachments_contracts",
        "question": "Show all contracts",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_attachments"],
    },
    # =========================================================================
    # NEW TOOL COVERAGE - Activity Search (Global)
    # =========================================================================
    {
        "id": "e2e_activities_meetings",
        "question": "What meetings do we have scheduled?",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_activities"],
    },
    {
        "id": "e2e_activities_calls",
        "question": "Show me all recent calls",
        "category": "data",
        "expected_mode": "data",
        "expected_tools": ["search_activities"],
    },
]


# =============================================================================
# Evaluation Functions
# =============================================================================

def run_e2e_test(
    test_case: dict,
    verbose: bool = False,
) -> E2EEvalResult:
    """Run a single end-to-end test case."""
    test_id = test_case["id"]
    question = test_case["question"]
    category = test_case["category"]
    expected_mode = test_case["expected_mode"]
    expected_tools = test_case.get("expected_tools", [])
    
    if verbose:
        console.print(f"\n  Testing: {test_id}")
        console.print(f"    Q: {question[:60]}...")
    
    start_time = time.time()
    error = None
    
    try:
        # Run the full agent pipeline
        result = answer_question(question)
        latency = (time.time() - start_time) * 1000
        
        answer = result.get("answer", "")
        sources = [s.get("id", "") for s in result.get("sources", [])]
        steps = result.get("steps", [])
        meta = result.get("meta", {})
        
        # Extract actual mode and tools from steps
        actual_mode = meta.get("mode", "unknown")
        actual_tools = []
        for step in steps:
            step_name = step.get("name", "")
            # Map step names to tool names
            if "company" in step_name.lower():
                actual_tools.append("company_lookup")
            elif "activit" in step_name.lower():
                actual_tools.append("recent_activity")
            elif "history" in step_name.lower():
                actual_tools.append("recent_history")
            elif "pipeline" in step_name.lower() or "opportunit" in step_name.lower():
                actual_tools.append("pipeline")
            elif "renewal" in step_name.lower():
                actual_tools.append("upcoming_renewals")
        
        # Remove duplicates
        actual_tools = list(dict.fromkeys(actual_tools))
        
    except Exception as e:
        error = str(e)
        latency = (time.time() - start_time) * 1000
        return E2EEvalResult(
            test_case_id=test_id,
            question=question,
            category=category,
            expected_mode=expected_mode,
            actual_mode="error",
            expected_tools=expected_tools,
            actual_tools=[],
            answer="",
            answer_relevance=0,
            answer_grounded=0,
            tool_selection_correct=False,
            has_sources=False,
            latency_ms=latency,
            total_tokens=0,
            error=error,
        )
    
    # Judge the response
    judge_result = judge_e2e_response(question, answer, sources)
    
    # Check tool selection correctness
    # Tools are "correct" if expected tools are subset of actual (may call more)
    tool_selection_correct = all(t in actual_tools for t in expected_tools)
    
    if verbose:
        relevance = "✓" if judge_result["answer_relevance"] else "✗"
        grounded = "✓" if judge_result["answer_grounded"] else "✗"
        console.print(f"    Mode: {actual_mode}, Tools: {actual_tools}")
        console.print(f"    Relevance: {relevance}, Grounded: {grounded}")
    
    return E2EEvalResult(
        test_case_id=test_id,
        question=question,
        category=category,
        expected_mode=expected_mode,
        actual_mode=actual_mode,
        expected_tools=expected_tools,
        actual_tools=actual_tools,
        answer=answer[:500],  # Truncate for storage
        answer_relevance=judge_result["answer_relevance"],
        answer_grounded=judge_result["answer_grounded"],
        tool_selection_correct=tool_selection_correct,
        has_sources=len(sources) > 0,
        latency_ms=latency,
        total_tokens=meta.get("total_tokens", 0),
        error=error,
        judge_explanation=judge_result["explanation"],
    )


def run_e2e_eval(
    limit: Optional[int] = None,
    verbose: bool = False,
) -> tuple[list[E2EEvalResult], E2EEvalSummary]:
    """
    Run end-to-end evaluation.
    
    Args:
        limit: Limit number of tests to run
        verbose: Print detailed progress
        
    Returns:
        Tuple of (results list, summary)
    """
    console.print("\n[bold blue]═══ End-to-End Agent Evaluation ═══[/bold blue]\n")
    
    test_cases = E2E_TEST_CASES[:limit] if limit else E2E_TEST_CASES
    results = []
    
    for test_case in track(test_cases, description="Running E2E tests..."):
        result = run_e2e_test(test_case, verbose=verbose)
        results.append(result)
    
    # Compute summary
    total = len(results)
    
    relevance_rate = sum(r.answer_relevance for r in results) / total if total > 0 else 0
    groundedness_rate = sum(r.answer_grounded for r in results) / total if total > 0 else 0
    tool_accuracy = sum(1 for r in results if r.tool_selection_correct) / total if total > 0 else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    
    # Compute P95 latency
    latencies = sorted([r.latency_ms for r in results])
    p95_index = int(len(latencies) * 0.95) if latencies else 0
    p95_latency = latencies[min(p95_index, len(latencies) - 1)] if latencies else 0.0
    
    # Import SLO for latency check
    from backend.agent.eval.models import SLO_LATENCY_P95_MS
    latency_slo_pass = p95_latency <= SLO_LATENCY_P95_MS
    
    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = {
                "count": 0,
                "relevance_sum": 0,
                "grounded_sum": 0,
            }
        by_category[cat]["count"] += 1
        by_category[cat]["relevance_sum"] += r.answer_relevance
        by_category[cat]["grounded_sum"] += r.answer_grounded
    
    for cat in by_category:
        count = by_category[cat]["count"]
        by_category[cat]["relevance_rate"] = by_category[cat]["relevance_sum"] / count
        by_category[cat]["groundedness_rate"] = by_category[cat]["grounded_sum"] / count
    
    summary = E2EEvalSummary(
        total_tests=total,
        answer_relevance_rate=relevance_rate,
        groundedness_rate=groundedness_rate,
        tool_selection_accuracy=tool_accuracy,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        latency_slo_pass=latency_slo_pass,
        by_category=by_category,
    )
    
    return results, summary


def print_e2e_eval_results(results: list[E2EEvalResult], summary: E2EEvalSummary) -> None:
    """Print end-to-end evaluation results."""
    # Summary table
    table = Table(title="E2E Evaluation Summary", show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    table.add_row("Total Tests", str(summary.total_tests))
    
    rel_color = "green" if summary.answer_relevance_rate >= 0.9 else "yellow" if summary.answer_relevance_rate >= 0.7 else "red"
    table.add_row("Answer Relevance", f"[{rel_color}]{summary.answer_relevance_rate:.1%}[/{rel_color}]")
    
    ground_color = "green" if summary.groundedness_rate >= 0.9 else "yellow" if summary.groundedness_rate >= 0.7 else "red"
    table.add_row("Groundedness", f"[{ground_color}]{summary.groundedness_rate:.1%}[/{ground_color}]")
    
    tool_color = "green" if summary.tool_selection_accuracy >= 0.9 else "yellow" if summary.tool_selection_accuracy >= 0.7 else "red"
    table.add_row("Tool Selection", f"[{tool_color}]{summary.tool_selection_accuracy:.1%}[/{tool_color}]")
    
    table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    
    p95_color = "green" if summary.latency_slo_pass else "red"
    table.add_row("P95 Latency", f"[{p95_color}]{summary.p95_latency_ms:.0f}ms[/{p95_color}]")
    
    console.print(table)
    
    # By-category breakdown
    cat_table = Table(title="Results by Category", show_header=True)
    cat_table.add_column("Category")
    cat_table.add_column("Tests", justify="right")
    cat_table.add_column("Relevance", justify="right")
    cat_table.add_column("Grounded", justify="right")
    
    for cat, stats in sorted(summary.by_category.items()):
        rel_color = "green" if stats["relevance_rate"] >= 0.9 else "yellow"
        ground_color = "green" if stats["groundedness_rate"] >= 0.9 else "yellow"
        cat_table.add_row(
            cat,
            str(stats["count"]),
            f"[{rel_color}]{stats['relevance_rate']:.0%}[/{rel_color}]",
            f"[{ground_color}]{stats['groundedness_rate']:.0%}[/{ground_color}]",
        )
    
    console.print(cat_table)
    
    # Show issues
    issues = [r for r in results if r.answer_relevance == 0 or r.answer_grounded == 0]
    if issues:
        console.print("\n[yellow bold]Issues Found:[/yellow bold]")
        for r in issues[:5]:  # Show first 5
            console.print(f"\n  [{r.test_case_id}] {r.question[:50]}...")
            console.print(f"    Relevance: {r.answer_relevance}, Grounded: {r.answer_grounded}")
            console.print(f"    Judge: {r.judge_explanation[:100]}...")
            if r.error:
                console.print(f"    [red]Error: {r.error}[/red]")


# =============================================================================
# CLI
# =============================================================================

app = typer.Typer()


@app.command()
def main(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit tests to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run end-to-end agent evaluation."""
    results, summary = run_e2e_eval(limit=limit, verbose=verbose)
    print_e2e_eval_results(results, summary)
    
    # Overall pass/fail
    overall_pass = (
        summary.answer_relevance_rate >= 0.8 and
        summary.groundedness_rate >= 0.8
    )
    
    if overall_pass:
        console.print("\n[green bold]✓ PASS: E2E evaluation meets thresholds[/green bold]")
    else:
        console.print("\n[red bold]✗ FAIL: E2E evaluation below thresholds[/red bold]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
