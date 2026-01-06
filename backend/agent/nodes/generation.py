"""
LangGraph generation nodes for agent workflow.

Handles answer synthesis and follow-up suggestion generation.
"""

import logging
import time

from backend.agent.nodes.state import AgentState
from backend.agent.core.config import get_config
from backend.agent.nodes.support.memory import format_history_for_prompt
from backend.agent.nodes.support.formatters import (
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_contacts_section,
    format_groups_section,
    format_attachments_section,
    format_docs_section,
    format_account_context_section,
    format_conversation_history_section,
)
from backend.agent.llm.helpers import (
    generate_follow_up_suggestions,
    call_answer_chain,
    call_not_found_chain,
)


logger = logging.getLogger(__name__)


def answer_node(state: AgentState) -> AgentState:
    config = get_config()
    start_time = time.time()

    logger.info("[Answer] Synthesizing response with LCEL chain...")

    try:
        company_data = state.get("company_data")
        llm_latency = 0

        # Handle company not found case
        if company_data and not company_data.get("found"):
            close_matches = company_data.get("close_matches", [])[: config.max_close_matches]
            matches_text = (
                "\n".join([f"- {m.get('name')} ({m.get('company_id')})" for m in close_matches])
                or "No similar companies found."
            )

            # Use LCEL chain for company not found
            answer, llm_latency = call_not_found_chain(
                question=state["question"],
                query=company_data.get("query", "unknown"),
                matches=matches_text,
            )

        else:
            # Build context sections
            conversation_history_section = format_conversation_history_section(
                state.get("messages", [])
            )
            company_section = format_company_section(company_data)
            contacts_section = format_contacts_section(state.get("contacts_data"))
            activities_section = format_activities_section(state.get("activities_data"))
            history_section = format_history_section(state.get("history_data"))
            pipeline_section = format_pipeline_section(state.get("pipeline_data"))
            renewals_section = format_renewals_section(state.get("renewals_data"))
            groups_section = format_groups_section(state.get("groups_data"))
            attachments_section = format_attachments_section(state.get("attachments_data"))
            docs_section = format_docs_section(state.get("docs_answer", ""))
            account_context_section = format_account_context_section(
                state.get("account_context_answer", "")
            )

            # Use LCEL chain for answer synthesis
            answer, llm_latency = call_answer_chain(
                question=state["question"],
                conversation_history_section=conversation_history_section,
                company_section=company_section,
                contacts_section=contacts_section,
                activities_section=activities_section,
                history_section=history_section,
                pipeline_section=pipeline_section,
                renewals_section=renewals_section,
                groups_section=groups_section,
                attachments_section=attachments_section,
                docs_section=docs_section,
                account_context_section=account_context_section,
            )

        # Validate answer
        if not answer or not answer.strip():
            logger.warning("[Answer] LLM returned empty answer, using fallback")
            answer = "I apologize, but I wasn't able to generate a complete response. Please try rephrasing your question."

        total_latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[Answer] Synthesized in {total_latency_ms}ms (LLM: {llm_latency}ms)")

        # Update messages for conversation memory (persisted via LangGraph checkpoint)
        messages = list(state.get("messages", []))
        messages.append(
            {
                "role": "user",
                "content": state["question"],
                "company_id": state.get("resolved_company_id"),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "company_id": state.get("resolved_company_id"),
            }
        )

        return {
            "answer": answer,
            "messages": messages,  # Updated messages for next turn
            "answer_latency_ms": total_latency_ms,
            "llm_latency_ms": llm_latency,
            "steps": [
                {
                    "id": "answer",
                    "label": "Synthesizing answer",
                    "status": "done",
                    "latency_ms": total_latency_ms,
                }
            ],
        }

    except Exception as e:
        total_latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[Answer] Failed after {total_latency_ms}ms: {e}")
        error_answer = f"I encountered an error generating the answer: {str(e)}"

        # Still update messages for conversation continuity
        messages = list(state.get("messages", []))
        messages.append(
            {
                "role": "user",
                "content": state["question"],
                "company_id": state.get("resolved_company_id"),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": error_answer,
                "company_id": state.get("resolved_company_id"),
            }
        )

        return {
            "answer": error_answer,
            "messages": messages,
            "answer_latency_ms": total_latency_ms,
            "llm_latency_ms": 0,
            "error": str(e),
            "steps": [
                {
                    "id": "answer",
                    "label": "Synthesizing answer",
                    "status": "error",
                    "latency_ms": total_latency_ms,
                }
            ],
        }


def followup_node(state: AgentState) -> AgentState:
    config = get_config()
    start_time = time.time()

    if not config.enable_follow_up_suggestions:
        logger.info("[Followup] Suggestions disabled in config")
        return {"follow_up_suggestions": []}

    logger.info("[Followup] Generating suggestions...")

    # Format conversation history for follow-up context
    messages = state.get("messages", [])
    conversation_history = format_history_for_prompt(messages) if messages else ""

    # Extract company name from company_data
    company_data = state.get("company_data", {})
    company_name = None
    if company_data and company_data.get("found"):
        company_info = company_data.get("company", {})
        company_name = company_info.get("name")

    # Build available data counts from raw_data
    raw_data = state.get("raw_data", {})
    available_data = {
        "contacts": len(raw_data.get("contacts", [])) if isinstance(raw_data, dict) else 0,
        "activities": len(raw_data.get("activities", [])) if isinstance(raw_data, dict) else 0,
        "opportunities": len(raw_data.get("opportunities", []))
        if isinstance(raw_data, dict)
        else 0,
        "history": len(raw_data.get("history", [])) if isinstance(raw_data, dict) else 0,
        "renewals": len(raw_data.get("renewals", [])) if isinstance(raw_data, dict) else 0,
        "pipeline_summary": raw_data.get("pipeline_summary")
        if isinstance(raw_data, dict)
        else None,
        "docs": len(state.get("doc_sources", [])),
    }

    try:
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            mode=state.get("mode_used", "auto"),
            company_id=state.get("resolved_company_id"),
            company_name=company_name,
            conversation_history=conversation_history,
            available_data=available_data,
        )

        # Validate and limit suggestions
        if suggestions:
            suggestions = [s for s in suggestions if s and s.strip()]
            suggestions = suggestions[: config.max_followup_suggestions]

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[Followup] Generated {len(suggestions)} suggestions in {latency_ms}ms")

        return {
            "follow_up_suggestions": suggestions,
            "followup_latency_ms": latency_ms,
            "steps": [
                {
                    "id": "followup",
                    "label": "Generating suggestions",
                    "status": "done",
                    "latency_ms": latency_ms,
                }
            ],
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"[Followup] Failed after {latency_ms}ms: {e}")
        return {
            "follow_up_suggestions": [],
            "followup_latency_ms": latency_ms,
            "steps": [
                {
                    "id": "followup",
                    "label": "Generating suggestions",
                    "status": "error",
                    "latency_ms": latency_ms,
                }
            ],
        }
