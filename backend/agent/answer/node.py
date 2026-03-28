"""Answer synthesis node for agent workflow."""

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.answer.answerer import call_answer_chain
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)

# Max iterations for Fetch→Answer loop
MAX_LOOP_COUNT = 2

# Patterns indicating answer needs more data
_MISSING_DATA_PATTERNS = [
    r"data not available",
    r"no (?:data|results|records) (?:found|available)",
    r"missing.*(?:field|column|table)",
    r"could not find",
    r"not present in (?:the )?(?:crm )?data",
]
_MISSING_DATA_REGEX = re.compile("|".join(_MISSING_DATA_PATTERNS), re.IGNORECASE)


def _detect_needs_more_data(answer: str, loop_count: int) -> bool:
    """Detect if answer indicates more data is needed.

    Returns True if:
    1. Answer contains "data not available" patterns
    2. We haven't exceeded max loop count
    3. The missing data seems fetchable (not just "question not answerable")
    """
    if loop_count >= MAX_LOOP_COUNT:
        logger.info(f"[Answer] Max loops ({MAX_LOOP_COUNT}) reached, not requesting more data")
        return False

    # Check for unfetchable patterns (don't loop for these)
    unfetchable = [
        "question not answerable",
        "out of scope",
        "cannot be answered",
        "not a crm question",
    ]
    answer_lower = answer.lower()
    if any(pattern in answer_lower for pattern in unfetchable):
        return False

    # Check for fetchable missing data
    if _MISSING_DATA_REGEX.search(answer):
        logger.info("[Answer] Detected missing data that may be fetchable")
        return True

    return False


def _generate_clarification() -> str:
    """Generate a clarification request."""
    return (
        "I'd be happy to help! Could you please provide more details about what "
        "you're looking for? For example:\n"
        "- Which company, contact, or deal are you asking about?\n"
        "- What specific information do you need (revenue, activities, pipeline)?\n"
        "- Any time period you're interested in?"
    )


def _generate_help_response() -> str:
    """Generate a help/capabilities response."""
    return (
        "I'm your CRM data assistant. I can help you with:\n\n"
        "**Data Queries**\n"
        "- Company information and renewals\n"
        "- Contact details and activities\n"
        "- Pipeline and opportunity analysis\n"
        "- Revenue and sales metrics\n\n"
        "**Example Questions**\n"
        "- \"Show me all deals closing this month\"\n"
        "- \"What's the total pipeline value?\"\n"
        "- \"List recent activities for Acme Corp\"\n\n"
        "Just ask a question about your CRM data and I'll find the answer!"
    )


def answer_node(state: AgentState) -> AgentState:
    """Synthesize answer from SQL results or handle clarify/help intents."""
    # Log immediately with flush to ensure visibility in Railway logs
    print("[Answer] Node entered", flush=True)

    intent = state.get("intent", "data_query")
    loop_count = state.get("loop_count", 0)

    logger.info(f"[Answer] Processing intent={intent}, loop={loop_count}")
    print(f"[Answer] Processing intent={intent}, loop={loop_count}", flush=True)

    # Handle non-data intents directly
    if intent == "clarify":
        answer = _generate_clarification()
        logger.info("[Answer] Generated clarification request")
        return {
            "answer": answer,
            "needs_more_data": False,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
        }

    if intent == "help":
        answer = _generate_help_response()
        logger.info("[Answer] Generated help response")
        return {
            "answer": answer,
            "needs_more_data": False,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
        }

    # DOCS intent: use pre-synthesized RAG answer with documentation sources
    if intent == "docs":
        sql_results = state.get("sql_results", {})
        rag_answer = sql_results.get("rag_answer")
        rag_sources = sql_results.get("rag_sources", [])

        if rag_answer:
            # Format answer with documentation sources
            source_refs = []
            for src in rag_sources[:3]:
                source_refs.append(f"- [{src['id']}] {src['source']}: \"{src['excerpt']}\"")

            if source_refs:
                answer = f"{rag_answer}\n\n**Documentation Sources:**\n" + "\n".join(source_refs)
            else:
                answer = rag_answer

            logger.info(f"[Answer] RAG response with {len(rag_sources)} sources")
        else:
            answer = (
                "I couldn't find relevant information in the Act! documentation. "
                "Try rephrasing your question or ask about specific features like "
                "importing contacts, creating groups, or using marketing automation."
            )
            logger.warning("[Answer] No RAG results found")

        return {
            "answer": answer,
            "needs_more_data": False,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
        }

    # DATA_QUERY intent: synthesize from SQL results
    try:
        raw_answer = call_answer_chain(
            question=state["question"],
            sql_results=state.get("sql_results", {}),
            conversation_history=format_conversation_for_prompt(state.get("messages", [])),
        )

        # Validate answer
        if not raw_answer or not raw_answer.strip():
            logger.warning("[Answer] LLM returned empty answer, using fallback")
            raw_answer = "I apologize, but I wasn't able to generate a complete response. Please try rephrasing your question."

        answer = raw_answer

        # Check if we need more data
        needs_more_data = _detect_needs_more_data(answer, loop_count)

        logger.info(f"[Answer] Synthesized response, needs_more_data={needs_more_data}")

        return {
            "answer": answer,
            "needs_more_data": needs_more_data,
            "loop_count": loop_count + 1,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
        }

    except Exception as e:
        logger.error(f"[Answer] Failed: {e}")
        error_answer = f"I encountered an error generating the answer: {str(e)}"

        return {
            "answer": error_answer,
            "needs_more_data": False,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=error_answer),
            ],
            "error": str(e),
        }


__all__ = ["answer_node"]
