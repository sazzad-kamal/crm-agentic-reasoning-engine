"""
Agent orchestration for answering CRM questions.

Coordinates:
1. Router - determine mode and extract parameters (LLM or heuristic)
2. Data tools - fetch CRM data
3. Docs RAG - fetch documentation (reuses backend.rag)
4. LLM - generate grounded answer

Enhanced features:
- Structured logging throughout
- Retry logic for LLM calls
- LLM-based routing option
- Dynamic progress tracking
- Audit logging
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from backend.agent.config import get_config, is_mock_mode
from backend.agent.schemas import (
    ChatResponse, Source, Step, RawData, MetaInfo, RouterResult
)
from backend.agent.llm_router import route_question
from backend.agent.audit import AgentAuditLogger
from backend.agent.tools import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
)
from backend.agent.datastore import get_datastore


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Progress Tracking
# =============================================================================

class StepStatus(str, Enum):
    """Status values for processing steps."""
    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class AgentProgress:
    """Tracks progress through the agent pipeline."""
    steps: list[Step] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def add_step(self, step_id: str, label: str, status: str = "done") -> None:
        """Add a completed step."""
        self.steps.append(Step(id=step_id, label=label, status=status))
        logger.debug(f"Step: {step_id} - {label} [{status}]")
    
    def get_elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int((time.time() - self.start_time) * 1000)
    
    def to_list(self) -> list[dict]:
        """Convert steps to list of dicts."""
        return [step.model_dump() for step in self.steps]


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

FOLLOW_UP_PROMPT = """Based on the user's question and the answer provided, suggest 3 natural follow-up questions they might want to ask next.

User's original question: {question}
Mode used: {mode}
Company context: {company}

Generate 3 SHORT, SPECIFIC follow-up questions that would be valuable. Focus on:
- Drilling deeper into the data shown
- Related information they might need
- Actionable next steps

Respond with ONLY a JSON array of 3 strings, nothing else:
["Question 1?", "Question 2?", "Question 3?"]"""


# =============================================================================
# LLM Helpers with Retry
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm(prompt: str, system_prompt: str) -> tuple[str, int]:
    """
    Call the LLM with retry logic for transient failures.
    
    Returns (response_text, latency_ms)
    """
    config = get_config()
    
    if is_mock_mode():
        logger.debug("Mock mode: Returning mock LLM response")
        return _mock_llm_response(prompt), 100
    
    # Import here to avoid loading OpenAI client when mocking
    from backend.common.llm_client import call_llm_with_metrics
    logger.debug(f"Calling LLM with model={config.llm_model}")
    
    result = call_llm_with_metrics(
        prompt=prompt,
        system_prompt=system_prompt,
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
        temperature=config.llm_temperature,
    )
    
    latency_ms = int(result["latency_ms"])
    logger.info(f"LLM response received in {latency_ms}ms")
    
    return result["response"], latency_ms


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


def _generate_follow_up_suggestions(
    question: str,
    mode: str,
    company_id: Optional[str] = None,
) -> list[str]:
    """
    Generate follow-up question suggestions using LLM.
    
    Returns list of 3 suggested follow-up questions.
    """
    config = get_config()
    
    if not config.enable_follow_up_suggestions:
        return []
    
    if is_mock_mode():
        # Return context-aware mock suggestions
        if "renewal" in question.lower():
            return [
                "Which renewals are at risk?",
                "What's the total renewal value this quarter?",
                "Show me accounts with no recent activity",
            ]
        elif "pipeline" in question.lower():
            return [
                "Which deals are stalled?",
                "What's the forecast for this quarter?",
                "Show me deals closing this month",
            ]
        else:
            return [
                "What are the recent activities?",
                "Show me the open opportunities",
                "Any upcoming renewals?",
            ]
    
    try:
        from backend.common.llm_client import call_llm
        
        prompt = FOLLOW_UP_PROMPT.format(
            question=question,
            mode=mode,
            company=company_id or "None specified",
        )
        
        response = call_llm(
            prompt=prompt,
            system_prompt="You are a helpful CRM assistant.",
            model=config.llm_model,
            temperature=0.7,  # Slightly creative for varied suggestions
            max_tokens=150,
        )
        
        # Parse JSON array from response
        import json
        text = response.strip()
        if text.startswith("["):
            suggestions = json.loads(text)
            if isinstance(suggestions, list) and len(suggestions) >= 3:
                return suggestions[:3]
        
        logger.warning(f"Failed to parse follow-up suggestions: {text[:100]}")
        return []
        
    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return []


# =============================================================================
# RAG Integration
# =============================================================================

def _call_docs_rag(question: str) -> tuple[str, list[Source]]:
    """
    Call the docs RAG pipeline from backend.rag.
    
    Returns (answer_text, doc_sources)
    """
    if is_mock_mode():
        return (
            "According to the documentation, you can find this feature "
            "in the Settings menu under Account Configuration.",
            [Source(type="doc", id="product_acme_crm_overview", label="Product Overview")]
        )
    
    try:
        from backend.rag.retrieval import create_backend
        from backend.rag.pipeline import answer_question as rag_answer
        
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
        logger.warning(f"Docs RAG failed: {e}")
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
# Data Gathering Helpers
# =============================================================================

def _gather_renewals_data(
    days: int,
    sources: list[Source],
    raw_data: dict,
) -> dict:
    """
    Gather renewal data when no specific company is targeted.
    
    Args:
        days: Number of days to look ahead for renewals
        sources: List to append sources to (modified in place)
        raw_data: Dict to update with renewal data (modified in place)
        
    Returns:
        The renewals result data dict
    """
    logger.debug(f"Fetching renewals for next {days} days")
    renewals_result = tool_upcoming_renewals(days=days)
    sources.extend(renewals_result.sources)
    raw_data["renewals"] = renewals_result.data.get("renewals", [])[:8]
    return renewals_result.data


def _gather_company_data(
    query: str,
    days: int,
    sources: list[Source],
    raw_data: dict,
) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict], Optional[str]]:
    """
    Gather all data for a specific company.
    
    This function fetches company info, activities, history, and pipeline
    data in sequence, updating the sources and raw_data as it goes.
    
    Args:
        query: Company ID or name to look up
        days: Number of days of history to fetch
        sources: List to append sources to (modified in place)
        raw_data: Dict to update with company data (modified in place)
        
    Returns:
        Tuple of (company_data, activities_data, history_data, pipeline_data, resolved_company_id)
    """
    logger.debug(f"Looking up company: {query}")
    company_result = tool_company_lookup(query)
    
    if not company_result.data.get("found"):
        # Company not found - return early with partial data
        return company_result.data, None, None, None, None
    
    # Company found - gather all related data
    company_data = company_result.data
    sources.extend(company_result.sources)
    resolved_company_id = company_data["company"]["company_id"]
    raw_data["companies"] = [company_data["company"]]
    
    logger.debug(f"Fetching data for company {resolved_company_id}")
    
    # Fetch activities (recent interactions)
    activities_result = tool_recent_activity(resolved_company_id, days=days)
    activities_data = activities_result.data
    sources.extend(activities_result.sources)
    raw_data["activities"] = activities_data.get("activities", [])[:8]
    
    # Fetch history (communications log)
    history_result = tool_recent_history(resolved_company_id, days=days)
    history_data = history_result.data
    sources.extend(history_result.sources)
    raw_data["history"] = history_data.get("history", [])[:8]
    
    # Fetch pipeline (open opportunities)
    pipeline_result = tool_pipeline(resolved_company_id)
    pipeline_data = pipeline_result.data
    sources.extend(pipeline_result.sources)
    raw_data["opportunities"] = pipeline_data.get("opportunities", [])[:8]
    raw_data["pipeline_summary"] = pipeline_data.get("summary")
    
    logger.info(
        f"Data fetched: activities={len(activities_data.get('activities', []))}, "
        f"history={len(history_data.get('history', []))}, "
        f"opps={len(pipeline_data.get('opportunities', []))}"
    )
    
    return company_data, activities_data, history_data, pipeline_data, resolved_company_id


def _generate_answer(
    question: str,
    company_data: Optional[dict],
    activities_data: Optional[dict],
    history_data: Optional[dict],
    pipeline_data: Optional[dict],
    renewals_data: Optional[dict],
    docs_answer: str,
) -> tuple[str, int]:
    """
    Generate the final answer using the LLM.
    
    Handles two cases:
    1. Company not found - asks for clarification with close matches
    2. Normal case - synthesizes answer from all gathered data
    
    Args:
        question: Original user question
        company_data: Company info dict (may indicate not found)
        activities_data: Recent activities or None
        history_data: Communication history or None
        pipeline_data: Open opportunities or None
        renewals_data: Upcoming renewals or None
        docs_answer: Documentation RAG answer or empty string
        
    Returns:
        Tuple of (answer_text, llm_latency_ms)
    """
    # Case 1: Company was searched but not found
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
        return _call_llm(prompt, AGENT_SYSTEM_PROMPT)
    
    # Case 2: Normal answer generation with available data
    prompt = DATA_ANSWER_PROMPT.format(
        question=question,
        company_section=_format_company_section(company_data),
        activities_section=_format_activities_section(activities_data),
        history_section=_format_history_section(history_data),
        pipeline_section=_format_pipeline_section(pipeline_data),
        renewals_section=_format_renewals_section(renewals_data),
        docs_section=_format_docs_section(docs_answer),
    )
    return _call_llm(prompt, AGENT_SYSTEM_PROMPT)


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
    config = get_config()
    progress = AgentProgress()
    audit = AgentAuditLogger()
    
    logger.info(f"Processing question: {question[:100]}...")
    
    sources: list[Source] = []
    raw_data = {
        "companies": [],
        "activities": [],
        "opportunities": [],
        "history": [],
        "renewals": [],
        "pipeline_summary": None,
    }
    
    # -------------------------------------------------------------------------
    # Step 1: Router (LLM or heuristic based on config)
    # -------------------------------------------------------------------------
    progress.add_step("router", "Understanding your question")
    
    try:
        router_result = route_question(question, mode=mode, company_id=company_id)
        logger.info(
            f"Routing complete: mode={router_result.mode_used}, "
            f"company={router_result.company_id}, intent={router_result.intent}"
        )
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        progress.add_step("router_error", f"Routing failed: {e}", status="error")
        return _build_error_response(
            f"Failed to understand question: {e}",
            progress, sources, raw_data, "unknown"
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
        progress.add_step("data", "Fetching CRM data")
        
        try:
            # Handle renewals intent (no specific company)
            if intent == "renewals" and not resolved_company_id:
                logger.debug(f"Fetching renewals for next {days} days")
                renewals_result = tool_upcoming_renewals(days=days)
                renewals_data = renewals_result.data
                sources.extend(renewals_result.sources)
                raw_data["renewals"] = renewals_data.get("renewals", [])[:8]
            
            # Handle company-specific queries
            elif resolved_company_id or router_result.company_name_query:
                # Lookup company
                query = resolved_company_id or router_result.company_name_query
                logger.debug(f"Looking up company: {query}")
                company_result = tool_company_lookup(query or "")
                
                if company_result.data.get("found"):
                    company_data = company_result.data
                    sources.extend(company_result.sources)
                    resolved_company_id = company_data["company"]["company_id"]
                    raw_data["companies"] = [company_data["company"]]
                    
                    logger.debug(f"Fetching data for company {resolved_company_id}")
                    
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
                    
                    logger.info(
                        f"Data fetched: activities={len(activities_data.get('activities', []))}, "
                        f"history={len(history_data.get('history', []))}, "
                        f"opps={len(pipeline_data.get('opportunities', []))}"
                    )
                
                else:
                    # Company not found - we'll handle this in the answer step
                    company_data = company_result.data
                    logger.info(f"Company not found: {query}")
            
            else:
                # No company specified - get general renewals
                logger.debug("No company specified, fetching general renewals")
                renewals_result = tool_upcoming_renewals(days=days)
                renewals_data = renewals_result.data
                sources.extend(renewals_result.sources)
                raw_data["renewals"] = renewals_data.get("renewals", [])[:8]
                
        except Exception as e:
            logger.error(f"Data gathering failed: {e}")
            progress.add_step("data_error", f"Error: {str(e)[:50]}", status="error")
    
    # -------------------------------------------------------------------------
    # Step 3: Docs RAG (if mode includes docs)
    # -------------------------------------------------------------------------
    docs_answer = ""
    docs_sources: list[Source] = []
    
    if "docs" in mode_used:
        progress.add_step("docs", "Checking documentation")
        
        try:
            docs_answer, docs_sources = _call_docs_rag(question)
            sources.extend(docs_sources)
            logger.info(f"Docs RAG returned {len(docs_sources)} sources")
        except Exception as e:
            logger.error(f"Docs RAG failed: {e}")
            progress.add_step("docs_error", f"Error: {str(e)[:50]}", status="error")
    else:
        progress.add_step("docs", "Skipped (data-only query)", status="skipped")
    
    # -------------------------------------------------------------------------
    # Step 4: Generate answer (LLM)
    # -------------------------------------------------------------------------
    progress.add_step("answer", "Synthesizing answer")
    
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
        
        logger.info(f"Answer synthesized in {llm_latency}ms")
    
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        progress.add_step("answer_error", f"Error: {str(e)[:50]}", status="error")
        answer = f"I encountered an error generating the answer: {str(e)}"
    
    # -------------------------------------------------------------------------
    # Audit Logging
    # -------------------------------------------------------------------------
    if config.enable_audit_logging:
        audit.log_query(
            question=question,
            mode_used=mode_used,
            company_id=resolved_company_id,
            latency_ms=progress.get_elapsed_ms(),
            source_count=len(sources),
            user_id=user_id,
            session_id=session_id,
        )
    
    # -------------------------------------------------------------------------
    # Step 5: Generate follow-up suggestions (if enabled)
    # -------------------------------------------------------------------------
    follow_up_suggestions: list[str] = []
    
    if config.enable_follow_up_suggestions:
        progress.add_step("follow_ups", "Generating suggestions")
        try:
            follow_up_suggestions = _generate_follow_up_suggestions(
                question=question,
                mode=mode_used,
                company_id=resolved_company_id,
            )
            logger.debug(f"Generated {len(follow_up_suggestions)} follow-up suggestions")
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
            # Non-critical - continue without suggestions
    
    # -------------------------------------------------------------------------
    # Build response
    # -------------------------------------------------------------------------
    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "steps": progress.to_list(),
        "raw_data": raw_data,
        "follow_up_suggestions": follow_up_suggestions,
        "meta": {
            "mode_used": mode_used,
            "latency_ms": progress.get_elapsed_ms(),
            "company_id": resolved_company_id,
            "days": days,
        }
    }


def _build_error_response(
    error_msg: str,
    progress: AgentProgress,
    sources: list[Source],
    raw_data: dict,
    mode_used: str,
) -> dict:
    """Build an error response."""
    return {
        "answer": f"I'm sorry, I encountered an error: {error_msg}",
        "sources": [s.model_dump() for s in sources],
        "steps": progress.to_list(),
        "raw_data": raw_data,
        "follow_up_suggestions": [],
        "meta": {
            "mode_used": mode_used,
            "latency_ms": progress.get_elapsed_ms(),
        }
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Enable logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    # Enable mock mode for testing
    os.environ["MOCK_LLM"] = "1"
    
    print("Testing Agent (MOCK_LLM=1)")
    print("=" * 60)
    
    questions = [
        "What's going on with Acme Manufacturing in the last 90 days?",
        "Which accounts have upcoming renewals in the next 90 days?",
        "How do I create a new opportunity?",
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
        print(f"\nAnswer:\n{result['answer'][:300]}...")
