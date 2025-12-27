"""
LangGraph node functions for agent workflow.

Each function represents a node in the graph that processes state.
"""

import logging
from typing import Literal

from backend.agent.state import AgentState
from backend.agent.config import get_config
from backend.agent.schemas import Source
from backend.agent.llm_router import route_question
from backend.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    COMPANY_NOT_FOUND_PROMPT,
    DATA_ANSWER_PROMPT,
)
from backend.agent.formatters import (
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_docs_section,
)
from backend.agent.llm_helpers import (
    call_llm,
    call_docs_rag,
    generate_follow_up_suggestions,
)
from backend.agent.tools import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
    tool_search_contacts,
    tool_search_companies,
    tool_group_members,
    tool_list_groups,
    tool_search_attachments,
    tool_pipeline_summary,
    tool_search_activities,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Router Node
# =============================================================================

def route_node(state: AgentState) -> AgentState:
    """
    Router node: Determine mode and extract parameters.

    Uses LLM-based or heuristic routing based on config.
    """
    logger.info(f"[Route] Processing: {state['question'][:50]}...")

    try:
        router_result = route_question(
            state["question"],
            mode=state.get("mode", "auto"),
            company_id=state.get("company_id"),
        )

        logger.info(
            f"[Route] Result: mode={router_result.mode_used}, "
            f"company={router_result.company_id}, intent={router_result.intent}"
        )

        return {
            "router_result": router_result,
            "mode_used": router_result.mode_used,
            "resolved_company_id": router_result.company_id,
            "days": router_result.days,
            "intent": router_result.intent,
            "steps": [{"id": "router", "label": "Understanding your question", "status": "done"}],
        }

    except Exception as e:
        logger.error(f"[Route] Failed: {e}")
        return {
            "mode_used": "docs",  # Fallback to docs
            "days": 90,
            "intent": "general",
            "error": f"Routing failed: {e}",
            "steps": [{"id": "router", "label": "Understanding your question", "status": "error"}],
        }


# =============================================================================
# Data Node
# =============================================================================

def data_node(state: AgentState) -> AgentState:
    """
    Data node: Fetch CRM data based on router output.

    Handles multiple intents:
    - company_status: Full company profile with activities, history, pipeline
    - renewals: Upcoming renewals across all accounts
    - pipeline / pipeline_summary: Deals and forecast data
    - contact_lookup / contact_search: Contact information
    - company_search: Search companies by criteria
    - groups: Group membership and lists
    - attachments: File/document search
    - activities: Activity search
    """
    logger.info(f"[Data] Fetching CRM data for intent={state.get('intent')}")

    sources: list[Source] = []
    raw_data = {
        "companies": [],
        "contacts": [],
        "activities": [],
        "opportunities": [],
        "history": [],
        "renewals": [],
        "groups": [],
        "attachments": [],
        "pipeline_summary": None,
    }

    company_data = None
    activities_data = None
    history_data = None
    pipeline_data = None
    renewals_data = None
    contacts_data = None
    groups_data = None
    attachments_data = None

    intent = state.get("intent", "general")
    resolved_company_id = state.get("resolved_company_id")
    days = state.get("days", 90)
    question = state.get("question", "").lower()

    try:
        # Pipeline Summary (aggregate across all companies)
        if intent == "pipeline_summary":
            logger.debug("[Data] Fetching aggregate pipeline summary")
            summary_result = tool_pipeline_summary()
            pipeline_data = summary_result.data
            sources.extend(summary_result.sources)
            raw_data["pipeline_summary"] = {
                "total_count": pipeline_data.get("total_count"),
                "total_value": pipeline_data.get("total_value"),
                "by_stage": pipeline_data.get("by_stage", []),
            }
            raw_data["opportunities"] = pipeline_data.get("top_opportunities", [])[:8]

        # Renewals (can be company-specific or aggregate)
        elif intent == "renewals":
            if resolved_company_id:
                company_result = tool_company_lookup(resolved_company_id)
                if company_result.data.get("found"):
                    company_data = company_result.data
                    sources.extend(company_result.sources)
                    raw_data["companies"] = [company_data["company"]]

            logger.debug(f"[Data] Fetching renewals for next {days} days")
            renewals_result = tool_upcoming_renewals(days=days)
            renewals_data = renewals_result.data
            sources.extend(renewals_result.sources)
            raw_data["renewals"] = renewals_data.get("renewals", [])[:8]

        # Contact lookup or search
        elif intent in ("contact_lookup", "contact_search"):
            logger.debug("[Data] Handling contact query")
            role = _extract_role_from_question(question)

            if resolved_company_id:
                contacts_result = tool_search_contacts(company_id=resolved_company_id, role=role)
            else:
                contacts_result = tool_search_contacts(role=role)

            contacts_data = contacts_result.data
            sources.extend(contacts_result.sources)
            raw_data["contacts"] = contacts_data.get("contacts", [])[:10]

        # Company search
        elif intent == "company_search":
            logger.debug("[Data] Searching companies")
            segment, industry = _extract_company_criteria(question)
            companies_result = tool_search_companies(segment=segment, industry=industry)
            company_data = companies_result.data
            sources.extend(companies_result.sources)
            raw_data["companies"] = company_data.get("companies", [])[:10]

        # Groups
        elif intent == "groups":
            logger.debug("[Data] Handling groups query")
            group_id = _extract_group_id(question)

            if group_id:
                members_result = tool_group_members(group_id)
                groups_data = members_result.data
                sources.extend(members_result.sources)
                raw_data["groups"] = [{
                    "group_id": group_id,
                    "name": groups_data.get("group_name"),
                    "members": groups_data.get("members", [])[:10],
                }]
            else:
                groups_result = tool_list_groups()
                groups_data = groups_result.data
                sources.extend(groups_result.sources)
                raw_data["groups"] = groups_data.get("groups", [])[:10]

        # Attachments
        elif intent == "attachments":
            logger.debug("[Data] Searching attachments")
            query = _extract_attachment_query(question)
            attachments_result = tool_search_attachments(query=query, company_id=resolved_company_id)
            attachments_data = attachments_result.data
            sources.extend(attachments_result.sources)
            raw_data["attachments"] = attachments_data.get("attachments", [])[:10]

        # Activities (search across all companies)
        elif intent == "activities" and not resolved_company_id:
            logger.debug("[Data] Searching activities across all companies")
            activity_type = _extract_activity_type(question)
            activities_result = tool_search_activities(activity_type=activity_type, days=days)
            activities_data = activities_result.data
            sources.extend(activities_result.sources)
            raw_data["activities"] = activities_data.get("activities", [])[:10]

        # Company-specific queries (default for company_status, pipeline, etc.)
        elif resolved_company_id or (state.get("router_result") and state.get("router_result").company_name_query):
            raw_data, sources, company_data, activities_data, history_data, pipeline_data, resolved_company_id = (
                _fetch_company_data(state, resolved_company_id, days, raw_data, sources)
            )

        # Fallback: General renewals query
        else:
            logger.debug("[Data] No specific intent, fetching general renewals")
            renewals_result = tool_upcoming_renewals(days=days)
            renewals_data = renewals_result.data
            sources.extend(renewals_result.sources)
            raw_data["renewals"] = renewals_data.get("renewals", [])[:8]

        return {
            "company_data": company_data,
            "activities_data": activities_data,
            "history_data": history_data,
            "pipeline_data": pipeline_data,
            "renewals_data": renewals_data,
            "contacts_data": contacts_data,
            "groups_data": groups_data,
            "attachments_data": attachments_data,
            "resolved_company_id": resolved_company_id,
            "sources": sources,
            "raw_data": raw_data,
            "steps": [{"id": "data", "label": "Fetching CRM data", "status": "done"}],
        }

    except Exception as e:
        logger.error(f"[Data] Failed: {e}")
        return {
            "raw_data": raw_data,
            "steps": [{"id": "data", "label": "Fetching CRM data", "status": "error"}],
            "error": f"Data fetch failed: {e}",
        }


def _extract_role_from_question(question: str) -> str | None:
    """Extract contact role from question."""
    if any(word in question for word in ["decision", "maker", "decision-maker"]):
        return "Decision Maker"
    elif "champion" in question:
        return "Champion"
    elif "executive" in question or "vp" in question or "director" in question:
        return "Executive"
    return None


def _extract_company_criteria(question: str) -> tuple[str | None, str | None]:
    """Extract segment and industry from question."""
    segment = None
    if "enterprise" in question:
        segment = "Enterprise"
    elif "smb" in question:
        segment = "SMB"
    elif "mid-market" in question or "midmarket" in question:
        segment = "Mid-Market"

    industry = None
    industries = ["software", "manufacturing", "healthcare", "food", "consulting", "retail"]
    for ind in industries:
        if ind in question:
            industry = ind.capitalize()
            break

    return segment, industry


def _extract_group_id(question: str) -> str | None:
    """Extract group ID from question keywords."""
    group_keywords = {
        "at risk": "GRP-AT-RISK",
        "at-risk": "GRP-AT-RISK",
        "champion": "GRP-CHAMPIONS",
        "churned": "GRP-CHURNED",
        "dormant": "GRP-DORMANT",
        "hot lead": "GRP-HOT-LEADS",
    }
    for keyword, gid in group_keywords.items():
        if keyword in question:
            return gid
    return None


def _extract_attachment_query(question: str) -> str | None:
    """Extract attachment search query from question."""
    search_terms = []
    attachment_words = ["proposal", "contract", "document", "agreement", "pdf", "report"]
    for word in attachment_words:
        if word in question:
            search_terms.append(word)
    return " ".join(search_terms) if search_terms else None


def _extract_activity_type(question: str) -> str | None:
    """Extract activity type from question."""
    if "call" in question:
        return "Call"
    elif "email" in question:
        return "Email"
    elif "meeting" in question:
        return "Meeting"
    elif "task" in question:
        return "Task"
    return None


def _fetch_company_data(state: AgentState, resolved_company_id: str | None, days: int,
                        raw_data: dict, sources: list[Source]) -> tuple:
    """Fetch full company data including activities, history, and pipeline."""
    query = resolved_company_id or (state.get("router_result").company_name_query if state.get("router_result") else None)
    logger.debug(f"[Data] Looking up company: {query}")

    company_result = tool_company_lookup(query or "")
    company_data = None
    activities_data = None
    history_data = None
    pipeline_data = None

    if company_result.data.get("found"):
        company_data = company_result.data
        sources.extend(company_result.sources)
        resolved_company_id = company_data["company"]["company_id"]
        raw_data["companies"] = [company_data["company"]]

        logger.debug(f"[Data] Fetching data for {resolved_company_id}")

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
            f"[Data] Fetched: activities={len(activities_data.get('activities', []))}, "
            f"history={len(history_data.get('history', []))}, "
            f"opps={len(pipeline_data.get('opportunities', []))}"
        )
    else:
        company_data = company_result.data
        logger.info(f"[Data] Company not found: {query}")

    return raw_data, sources, company_data, activities_data, history_data, pipeline_data, resolved_company_id


# =============================================================================
# Docs Node
# =============================================================================

def docs_node(state: AgentState) -> AgentState:
    """
    Docs node: Fetch documentation via RAG.
    """
    logger.info("[Docs] Querying documentation...")

    try:
        docs_answer, docs_sources = call_docs_rag(state["question"])
        logger.info(f"[Docs] Retrieved {len(docs_sources)} sources")

        return {
            "docs_answer": docs_answer,
            "docs_sources": docs_sources,
            "sources": docs_sources,
            "steps": [{"id": "docs", "label": "Checking documentation", "status": "done"}],
        }

    except Exception as e:
        logger.error(f"[Docs] Failed: {e}")
        return {
            "docs_answer": "",
            "docs_sources": [],
            "steps": [{"id": "docs", "label": "Checking documentation", "status": "error"}],
        }


# =============================================================================
# Skip Nodes
# =============================================================================

def skip_data_node(state: AgentState) -> AgentState:
    """Placeholder when data is skipped."""
    return {
        "raw_data": {
            "companies": [],
            "contacts": [],
            "activities": [],
            "opportunities": [],
            "history": [],
            "renewals": [],
            "groups": [],
            "attachments": [],
            "pipeline_summary": None,
        },
        "steps": [{"id": "data", "label": "Skipped (docs-only query)", "status": "skipped"}],
    }


def skip_docs_node(state: AgentState) -> AgentState:
    """Placeholder when docs is skipped."""
    return {
        "docs_answer": "",
        "docs_sources": [],
        "steps": [{"id": "docs", "label": "Skipped (data-only query)", "status": "skipped"}],
    }


# =============================================================================
# Answer Node
# =============================================================================

def answer_node(state: AgentState) -> AgentState:
    """
    Answer node: Synthesize final answer using LLM.
    """
    logger.info("[Answer] Synthesizing response...")

    try:
        company_data = state.get("company_data")

        # Handle company not found case
        if company_data and not company_data.get("found"):
            matches_text = "\n".join([
                f"- {m.get('name')} ({m.get('company_id')})"
                for m in company_data.get("close_matches", [])[:5]
            ]) or "No similar companies found."

            prompt = COMPANY_NOT_FOUND_PROMPT.format(
                question=state["question"],
                query=company_data.get("query", "unknown"),
                matches=matches_text,
            )

            answer, llm_latency = call_llm(prompt, AGENT_SYSTEM_PROMPT)

        else:
            # Build context sections
            company_section = format_company_section(company_data)
            activities_section = format_activities_section(state.get("activities_data"))
            history_section = format_history_section(state.get("history_data"))
            pipeline_section = format_pipeline_section(state.get("pipeline_data"))
            renewals_section = format_renewals_section(state.get("renewals_data"))
            docs_section = format_docs_section(state.get("docs_answer", ""))

            prompt = DATA_ANSWER_PROMPT.format(
                question=state["question"],
                company_section=company_section,
                activities_section=activities_section,
                history_section=history_section,
                pipeline_section=pipeline_section,
                renewals_section=renewals_section,
                docs_section=docs_section,
            )

            answer, llm_latency = call_llm(prompt, AGENT_SYSTEM_PROMPT)

        logger.info(f"[Answer] Synthesized in {llm_latency}ms")

        return {
            "answer": answer,
            "steps": [{"id": "answer", "label": "Synthesizing answer", "status": "done"}],
        }

    except Exception as e:
        logger.error(f"[Answer] Failed: {e}")
        return {
            "answer": f"I encountered an error generating the answer: {str(e)}",
            "steps": [{"id": "answer", "label": "Synthesizing answer", "status": "error"}],
        }


# =============================================================================
# Follow-up Node
# =============================================================================

def followup_node(state: AgentState) -> AgentState:
    """
    Follow-up node: Generate suggested follow-up questions.
    """
    config = get_config()

    if not config.enable_follow_up_suggestions:
        return {"follow_up_suggestions": []}

    logger.info("[Followup] Generating suggestions...")

    try:
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            mode=state.get("mode_used", "auto"),
            company_id=state.get("resolved_company_id"),
        )
        logger.debug(f"[Followup] Generated {len(suggestions)} suggestions")

        return {
            "follow_up_suggestions": suggestions,
            "steps": [{"id": "followup", "label": "Generating suggestions", "status": "done"}],
        }

    except Exception as e:
        logger.warning(f"[Followup] Failed: {e}")
        return {
            "follow_up_suggestions": [],
            "steps": [{"id": "followup", "label": "Generating suggestions", "status": "error"}],
        }


# =============================================================================
# Routing Logic
# =============================================================================

def route_by_mode(state: AgentState) -> Literal["data_only", "docs_only", "data_and_docs"]:
    """
    Conditional edge: Route based on mode_used.
    """
    mode = state.get("mode_used", "data+docs")

    if mode == "data":
        return "data_only"
    elif mode == "docs":
        return "docs_only"
    else:  # "data+docs" or fallback
        return "data_and_docs"


__all__ = [
    "route_node",
    "data_node",
    "docs_node",
    "skip_data_node",
    "skip_docs_node",
    "answer_node",
    "followup_node",
    "route_by_mode",
]
