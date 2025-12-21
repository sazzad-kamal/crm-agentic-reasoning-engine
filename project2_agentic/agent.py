"""
Agent orchestration for answering CRM questions.

Coordinates:
1. Router - determine mode and extract parameters
2. Data tools - fetch CRM data
3. Docs RAG - fetch documentation (reuses project1_rag)
4. LLM - generate grounded answer
"""

import os
import time
from typing import Optional

from project2_agentic.schemas import (
    ChatResponse, Source, Step, RawData, MetaInfo, RouterResult
)
from project2_agentic.router import route_question
from project2_agentic.tools import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
)
from project2_agentic.datastore import get_datastore


# =============================================================================
# Configuration
# =============================================================================

# Check for mock mode (for testing without API key)
MOCK_LLM = os.environ.get("MOCK_LLM", "0") == "1"


# =============================================================================
# Prompts
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are a helpful CRM assistant for Acme CRM Suite. 
Your job is to answer questions about company accounts, activities, pipeline, and renewals using ONLY the provided CRM data.

IMPORTANT RULES:
1. Use ONLY the provided data to answer. Do not make up information.
2. Be concise and high-signal - busy sales professionals are reading this.
3. Structure your response as:
   - A brief summary (1-2 sentences)
   - Key facts as bullet points
   - 2-3 suggested next actions (when relevant)
4. If company data was provided, always mention the company name.
5. If documentation excerpts are provided, incorporate relevant guidance.
6. If a company was not found, ask a clarifying question and list close matches.
7. Format currency values with $ and commas.
8. Format dates in a human-readable way (e.g., "March 31, 2026").

Keep your answer SHORT and ACTIONABLE."""

COMPANY_NOT_FOUND_PROMPT = """The user asked about a company but we couldn't find an exact match.

User's question: {question}
Search query: {query}

Close matches found:
{matches}

Please respond with:
1. Acknowledge we couldn't find an exact match
2. Ask a clarifying question
3. List the close matches so they can clarify"""

DATA_ANSWER_PROMPT = """Based on the following CRM data, answer the user's question.

User's question: {question}

{company_section}

{activities_section}

{history_section}

{pipeline_section}

{renewals_section}

{docs_section}

Please provide a helpful, grounded response following the rules in your system prompt."""


# =============================================================================
# LLM Helpers
# =============================================================================

def _call_llm(prompt: str, system_prompt: str) -> tuple[str, int]:
    """
    Call the LLM with mock support for testing.
    
    Returns (response_text, latency_ms)
    """
    if MOCK_LLM:
        return _mock_llm_response(prompt), 100
    
    # Import here to avoid loading OpenAI client when mocking
    from shared.llm_client import call_llm_with_metrics
    
    result = call_llm_with_metrics(
        prompt=prompt,
        system_prompt=system_prompt,
        model="gpt-4.1-mini",
        max_tokens=800,
        temperature=0.1,
    )
    
    return result["response"], int(result["latency_ms"])


def _mock_llm_response(prompt: str) -> str:
    """Generate a mock response for testing."""
    # Extract some context from the prompt for a semi-realistic response
    if "couldn't find an exact match" in prompt:
        return (
            "I couldn't find an exact match for that company in the CRM. "
            "Could you clarify which company you're asking about? "
            "Here are some similar companies I found that might be what you're looking for."
        )
    
    if "renewal" in prompt.lower():
        return (
            "**Upcoming Renewals Summary**\n\n"
            "Based on the CRM data, here are the accounts with upcoming renewals:\n\n"
            "• Several accounts have renewals coming up in the specified timeframe\n"
            "• Review each account's health status before the renewal date\n\n"
            "**Suggested Actions:**\n"
            "1. Schedule check-in calls with at-risk accounts\n"
            "2. Prepare renewal proposals for key accounts\n"
            "3. Review recent activity levels to identify any concerns"
        )
    
    if "pipeline" in prompt.lower():
        return (
            "**Pipeline Summary**\n\n"
            "Here's the current pipeline status based on CRM data:\n\n"
            "• Open opportunities are progressing through various stages\n"
            "• Total pipeline value and deal count are shown in the data\n\n"
            "**Suggested Actions:**\n"
            "1. Focus on deals in Proposal and Negotiation stages\n"
            "2. Follow up on stalled opportunities\n"
            "3. Update expected close dates if needed"
        )
    
    # Default response for company status questions
    return (
        "**Account Summary**\n\n"
        "Based on the CRM data provided:\n\n"
        "• Recent activities show engagement with the account\n"
        "• Pipeline includes open opportunities in various stages\n"
        "• History log shows recent touchpoints\n\n"
        "**Suggested Actions:**\n"
        "1. Review recent activity and follow up if needed\n"
        "2. Check opportunity progress and update stages\n"
        "3. Confirm next steps with key contacts"
    )


# =============================================================================
# RAG Integration
# =============================================================================

def _call_docs_rag(question: str) -> tuple[str, list[Source]]:
    """
    Call the docs RAG pipeline from project1_rag.
    
    Returns (answer_text, doc_sources)
    """
    if MOCK_LLM:
        return (
            "According to the documentation, you can find this feature "
            "in the Settings menu under Account Configuration.",
            [Source(type="doc", id="product_acme_crm_overview", label="Product Overview")]
        )
    
    try:
        from project1_rag.retrieval_backend import create_backend
        from project1_rag.rag_pipeline import answer_question as rag_answer
        
        backend = create_backend()
        result = rag_answer(question, backend, use_hyde=True, use_rewrite=True)
        
        # Extract sources from used docs
        doc_sources = []
        for doc_id in result.get("doc_ids_used", [])[:3]:
            # Format doc_id into readable label
            label = doc_id.replace("_", " ").replace(".md", "").title()
            doc_sources.append(Source(type="doc", id=doc_id, label=label))
        
        return result.get("answer", ""), doc_sources
    
    except Exception as e:
        print(f"Warning: Docs RAG failed: {e}")
        return "", []


# =============================================================================
# Context Formatting
# =============================================================================

def _format_company_section(company_data: Optional[dict]) -> str:
    """Format company data for the prompt."""
    if not company_data:
        return ""
    
    company = company_data.get("company")
    if not company:
        return ""
    
    lines = [
        "=== COMPANY INFO ===",
        f"Name: {company.get('name', 'N/A')}",
        f"ID: {company.get('company_id', 'N/A')}",
        f"Status: {company.get('status', 'N/A')}",
        f"Plan: {company.get('plan', 'N/A')}",
        f"Industry: {company.get('industry', 'N/A')}",
        f"Region: {company.get('region', 'N/A')}",
        f"Account Owner: {company.get('account_owner', 'N/A')}",
        f"Renewal Date: {company.get('renewal_date', 'N/A')}",
        f"Health: {company.get('health_flags', 'N/A')}",
    ]
    
    # Add contacts if present
    contacts = company_data.get("contacts", [])
    if contacts:
        lines.append("\nKey Contacts:")
        for c in contacts[:3]:
            name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip()
            lines.append(f"  - {name} ({c.get('job_title', 'N/A')}): {c.get('email', 'N/A')}")
    
    return "\n".join(lines)


def _format_activities_section(activities_data: Optional[dict]) -> str:
    """Format activities data for the prompt."""
    if not activities_data:
        return ""
    
    activities = activities_data.get("activities", [])
    if not activities:
        return "=== RECENT ACTIVITIES ===\nNo recent activities found."
    
    lines = [
        f"=== RECENT ACTIVITIES ({activities_data.get('count', 0)} found, last {activities_data.get('days', 90)} days) ==="
    ]
    
    for act in activities[:8]:
        due = act.get("due_datetime", act.get("created_at", "N/A"))
        if due and "T" in str(due):
            due = str(due).split("T")[0]
        lines.append(
            f"- [{act.get('type', 'N/A')}] {act.get('subject', 'N/A')} "
            f"(Owner: {act.get('owner', 'N/A')}, Due: {due}, Status: {act.get('status', 'N/A')})"
        )
    
    return "\n".join(lines)


def _format_history_section(history_data: Optional[dict]) -> str:
    """Format history data for the prompt."""
    if not history_data:
        return ""
    
    history = history_data.get("history", [])
    if not history:
        return "=== HISTORY LOG ===\nNo recent history entries."
    
    lines = [
        f"=== HISTORY LOG ({history_data.get('count', 0)} entries, last {history_data.get('days', 90)} days) ==="
    ]
    
    for h in history[:8]:
        occurred = h.get("occurred_at", "N/A")
        if occurred and "T" in str(occurred):
            occurred = str(occurred).split("T")[0]
        lines.append(
            f"- [{h.get('type', 'N/A')}] {h.get('subject', 'N/A')} "
            f"(Date: {occurred}, Owner: {h.get('owner', 'N/A')})"
        )
        if h.get("description"):
            desc = str(h.get("description", ""))[:100]
            if len(str(h.get("description", ""))) > 100:
                desc += "..."
            lines.append(f"    Note: {desc}")
    
    return "\n".join(lines)


def _format_pipeline_section(pipeline_data: Optional[dict]) -> str:
    """Format pipeline data for the prompt."""
    if not pipeline_data:
        return ""
    
    summary = pipeline_data.get("summary", {})
    opps = pipeline_data.get("opportunities", [])
    
    if not summary.get("total_count"):
        return "=== PIPELINE ===\nNo open opportunities."
    
    lines = [
        f"=== PIPELINE SUMMARY ===",
        f"Total Open Deals: {summary.get('total_count', 0)}",
        f"Total Value: ${summary.get('total_value', 0):,.0f}",
        "\nBy Stage:"
    ]
    
    for stage, data in summary.get("stages", {}).items():
        lines.append(f"  - {stage}: {data.get('count', 0)} deals (${data.get('total_value', 0):,.0f})")
    
    if opps:
        lines.append("\nOpen Opportunities:")
        for opp in opps[:5]:
            close_date = opp.get("expected_close_date", "N/A")
            lines.append(
                f"  - {opp.get('name', 'N/A')}: {opp.get('stage', 'N/A')} - "
                f"${opp.get('value', 0):,} (Close: {close_date})"
            )
    
    return "\n".join(lines)


def _format_renewals_section(renewals_data: Optional[dict]) -> str:
    """Format renewals data for the prompt."""
    if not renewals_data:
        return ""
    
    renewals = renewals_data.get("renewals", [])
    if not renewals:
        return "=== UPCOMING RENEWALS ===\nNo renewals in the specified timeframe."
    
    lines = [
        f"=== UPCOMING RENEWALS ({renewals_data.get('count', 0)} accounts, next {renewals_data.get('days', 90)} days) ==="
    ]
    
    for r in renewals[:10]:
        lines.append(
            f"- {r.get('name', 'N/A')} ({r.get('company_id', 'N/A')}): "
            f"Renewal {r.get('renewal_date', 'N/A')} | Plan: {r.get('plan', 'N/A')} | "
            f"Health: {r.get('health_flags', 'N/A')}"
        )
    
    return "\n".join(lines)


def _format_docs_section(docs_answer: str) -> str:
    """Format docs RAG answer for the prompt."""
    if not docs_answer:
        return ""
    
    return f"=== DOCUMENTATION GUIDANCE ===\n{docs_answer}"


# =============================================================================
# Main Agent Function
# =============================================================================

def answer_question(
    question: str,
    mode: str = "auto",
    company_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Answer a CRM question using the agent pipeline.
    
    Args:
        question: The user's question
        mode: Mode override ("auto", "docs", "data", "data+docs")
        company_id: Pre-specified company ID
        session_id: Optional session ID (for future stateful use)
        user_id: Optional user ID (for future personalization)
        
    Returns:
        Dict matching ChatResponse schema:
        {
            "answer": str,
            "sources": list[Source],
            "steps": list[Step],
            "raw_data": RawData,
            "meta": MetaInfo
        }
    """
    start_time = time.time()
    
    steps = []
    sources = []
    raw_data = {
        "companies": [],
        "activities": [],
        "opportunities": [],
        "history": [],
        "renewals": [],
        "pipeline_summary": None,
    }
    
    # -------------------------------------------------------------------------
    # Step 1: Router
    # -------------------------------------------------------------------------
    steps.append(Step(id="router", label="Understanding your question", status="done"))
    
    try:
        router_result = route_question(question, mode=mode, company_id=company_id)
    except Exception as e:
        steps[-1].status = "error"
        return _build_error_response(
            f"Failed to understand question: {e}",
            steps, sources, raw_data, "unknown", start_time
        )
    
    mode_used = router_result.mode_used
    resolved_company_id = router_result.company_id
    days = router_result.days
    intent = router_result.intent
    
    # -------------------------------------------------------------------------
    # Step 2: Data gathering (if mode includes data)
    # -------------------------------------------------------------------------
    company_data = None
    activities_data = None
    history_data = None
    pipeline_data = None
    renewals_data = None
    
    if "data" in mode_used:
        steps.append(Step(id="data", label="Checking CRM records", status="done"))
        
        try:
            # Handle renewals intent (no specific company)
            if intent == "renewals" and not resolved_company_id:
                renewals_result = tool_upcoming_renewals(days=days)
                renewals_data = renewals_result.data
                sources.extend(renewals_result.sources)
                raw_data["renewals"] = renewals_data.get("renewals", [])[:8]
            
            # Handle company-specific queries
            elif resolved_company_id or router_result.company_name_query:
                # Lookup company
                query = resolved_company_id or router_result.company_name_query
                company_result = tool_company_lookup(query or "")
                
                if company_result.data.get("found"):
                    company_data = company_result.data
                    sources.extend(company_result.sources)
                    resolved_company_id = company_data["company"]["company_id"]
                    raw_data["companies"] = [company_data["company"]]
                    
                    # Get activities
                    activities_result = tool_recent_activity(resolved_company_id, days=days)
                    activities_data = activities_result.data
                    sources.extend(activities_result.sources)
                    raw_data["activities"] = activities_data.get("activities", [])[:8]
                    
                    # Get history
                    history_result = tool_recent_history(resolved_company_id, days=days)
                    history_data = history_result.data
                    sources.extend(history_result.sources)
                    raw_data["history"] = history_data.get("history", [])[:8]
                    
                    # Get pipeline
                    pipeline_result = tool_pipeline(resolved_company_id)
                    pipeline_data = pipeline_result.data
                    sources.extend(pipeline_result.sources)
                    raw_data["opportunities"] = pipeline_data.get("opportunities", [])[:8]
                    raw_data["pipeline_summary"] = pipeline_data.get("summary")
                
                else:
                    # Company not found - we'll handle this in the answer step
                    company_data = company_result.data
                    steps[-1].label = "Company not found in CRM"
            
            else:
                # No company specified - get general renewals
                renewals_result = tool_upcoming_renewals(days=days)
                renewals_data = renewals_result.data
                sources.extend(renewals_result.sources)
                raw_data["renewals"] = renewals_data.get("renewals", [])[:8]
                
        except Exception as e:
            steps[-1].status = "error"
            steps[-1].label = f"Error fetching CRM data: {str(e)[:50]}"
    
    # -------------------------------------------------------------------------
    # Step 3: Docs RAG (if mode includes docs)
    # -------------------------------------------------------------------------
    docs_answer = ""
    docs_sources = []
    
    if "docs" in mode_used:
        steps.append(Step(id="docs", label="Checking help docs", status="done"))
        
        try:
            docs_answer, docs_sources = _call_docs_rag(question)
            sources.extend(docs_sources)
        except Exception as e:
            steps[-1].status = "error"
            steps[-1].label = f"Error fetching docs: {str(e)[:50]}"
    else:
        steps.append(Step(id="docs", label="Skipping help docs (not needed)", status="done"))
    
    # -------------------------------------------------------------------------
    # Step 4: Generate answer
    # -------------------------------------------------------------------------
    steps.append(Step(id="answer", label="Drafting grounded answer", status="done"))
    
    try:
        # Handle company not found case
        if company_data and not company_data.get("found"):
            matches_text = "\n".join([
                f"- {m.get('name')} ({m.get('company_id')})"
                for m in company_data.get("close_matches", [])[:5]
            ]) or "No similar companies found."
            
            prompt = COMPANY_NOT_FOUND_PROMPT.format(
                question=question,
                query=company_data.get("query", "unknown"),
                matches=matches_text,
            )
            
            answer, llm_latency = _call_llm(prompt, AGENT_SYSTEM_PROMPT)
        
        else:
            # Build context sections
            company_section = _format_company_section(company_data)
            activities_section = _format_activities_section(activities_data)
            history_section = _format_history_section(history_data)
            pipeline_section = _format_pipeline_section(pipeline_data)
            renewals_section = _format_renewals_section(renewals_data)
            docs_section = _format_docs_section(docs_answer)
            
            prompt = DATA_ANSWER_PROMPT.format(
                question=question,
                company_section=company_section,
                activities_section=activities_section,
                history_section=history_section,
                pipeline_section=pipeline_section,
                renewals_section=renewals_section,
                docs_section=docs_section,
            )
            
            answer, llm_latency = _call_llm(prompt, AGENT_SYSTEM_PROMPT)
    
    except Exception as e:
        steps[-1].status = "error"
        answer = f"I encountered an error generating the answer: {str(e)}"
        llm_latency = 0
    
    # -------------------------------------------------------------------------
    # Build response
    # -------------------------------------------------------------------------
    total_latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "steps": [s.model_dump() for s in steps],
        "raw_data": raw_data,
        "meta": {
            "mode_used": mode_used,
            "latency_ms": total_latency_ms,
            "company_id": resolved_company_id,
            "days": days,
        }
    }


def _build_error_response(
    error_msg: str,
    steps: list[Step],
    sources: list[Source],
    mode_used: str,
    start_time: float
) -> dict:
    """Build an error response."""
    return {
        "answer": f"I'm sorry, I encountered an error: {error_msg}",
        "sources": [s.model_dump() for s in sources],
        "steps": [s.model_dump() for s in steps],
        "raw_data": {
            "companies": [],
            "activities": [],
            "opportunities": [],
            "history": [],
        },
        "meta": {
            "mode_used": mode_used,
            "latency_ms": int((time.time() - start_time) * 1000),
        }
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Enable mock mode for testing
    os.environ["MOCK_LLM"] = "1"
    
    print("Testing Agent (MOCK_LLM=1)")
    print("=" * 60)
    
    questions = [
        "What's going on with Acme Manufacturing in the last 90 days?",
        "Which accounts have upcoming renewals in the next 90 days?",
        "Show the open pipeline for Beta Tech Solutions",
    ]
    
    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print("-" * 60)
        
        result = answer_question(q)
        
        print(f"Mode: {result['meta']['mode_used']}")
        print(f"Company: {result['meta'].get('company_id')}")
        print(f"Latency: {result['meta']['latency_ms']}ms")
        print(f"\nSteps:")
        for step in result['steps']:
            print(f"  - {step['id']}: {step['label']} [{step['status']}]")
        print(f"\nSources ({len(result['sources'])}):")
        for src in result['sources'][:3]:
            print(f"  - {src['type']}: {src['label']}")
        print(f"\nRaw Data Counts:")
        print(f"  - Companies: {len(result['raw_data'].get('companies', []))}")
        print(f"  - Activities: {len(result['raw_data'].get('activities', []))}")
        print(f"  - Opportunities: {len(result['raw_data'].get('opportunities', []))}")
        print(f"  - Renewals: {len(result['raw_data'].get('renewals', []))}")
        print(f"\nAnswer:\n{result['answer'][:300]}...")
