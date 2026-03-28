"""Intent classifier for Supervisor routing decisions."""

import logging
from enum import Enum

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Possible intents for user questions."""

    DATA_QUERY = "data_query"  # Needs SQL/data lookup
    COMPARE = "compare"        # Comparison queries (A vs B)
    TREND = "trend"            # Time-series/trend analysis
    COMPLEX = "complex"        # Multi-part queries needing planning
    EXPORT = "export"          # Export data to CSV/PDF
    HEALTH = "health"          # Account health score calculation
    DOCS = "docs"              # Documentation/how-to questions (RAG)
    GRAPH = "graph"            # Multi-hop relationship queries (Neo4j)
    CLARIFY = "clarify"        # Question is unclear, ask for clarification
    HELP = "help"              # General help, no data needed


_CLASSIFIER_PROMPT = """You are an intent classifier for a CRM data assistant.

Classify the user's question into exactly ONE of these categories:

1. DOCS - User asks HOW TO do something in Act! CRM, or wants product documentation
   Examples: "How do I import contacts?", "What's the keyboard shortcut for...", "How to create a group?"

2. COMPARE - User wants to compare two things (e.g., time periods, accounts, products)
   Examples: "Compare Q1 vs Q2 revenue", "How does Acme compare to TechCorp?", "Sales this year vs last year"

3. TREND - User wants time-series analysis or trend information
   Examples: "Show revenue trend over time", "Monthly growth rate", "How has pipeline changed?"

4. COMPLEX - User asks a multi-part question requiring multiple data queries
   Examples: "Show deals and compare Q1 vs Q2", "List companies and their health scores", "Revenue by region and trends"

5. EXPORT - User wants to export or download data
   Examples: "Export deals to CSV", "Download report", "Generate PDF of contacts"

6. HEALTH - User wants account/company health score or analysis
   Examples: "What's Acme's health score?", "Show account health", "Which accounts need attention?"

7. DATA_QUERY - User wants specific data from the CRM database (simple query)
   Examples: "Show me Q1 revenue", "List all contacts at Acme", "What deals close this month?"

8. GRAPH - User asks about multi-hop relationships between entities (contacts, companies, deals)
   Examples: "Which contacts at at-risk companies have deals closing?", "Show the relationship chain for Acme", "Who is connected to companies with open renewals?"

9. CLARIFY - Question is too vague or ambiguous to answer
   Examples: "the thing", "show me", "what about it?", "yes", "that one"

10. HELP - User wants help using the system or asks general questions (no data needed)
   Examples: "what can you do?", "help", "how does this work?", "hello"

Conversation history:
{history}

User question: {question}

Respond with ONLY one word: DOCS, COMPARE, TREND, COMPLEX, EXPORT, HEALTH, DATA_QUERY, GRAPH, CLARIFY, or HELP"""


def classify_intent(question: str, conversation_history: str = "") -> Intent:
    """Classify user question into an intent category.

    Args:
        question: The user's question
        conversation_history: Previous conversation context

    Returns:
        Intent enum value
    """
    # Quick heuristics for obvious cases (saves API call)
    q_lower = question.lower().strip()

    # Very short or contextual phrases need clarification
    if len(q_lower) < 4 or q_lower in {"yes", "no", "ok", "sure", "that", "this", "it"}:
        logger.info(f"[Supervisor] Quick classify '{question}' -> CLARIFY (too short)")
        return Intent.CLARIFY

    # Help phrases - system capabilities (check FIRST before DOCS)
    help_phrases_priority = {
        "what can you do", "how does this work", "how does it work",
        "what are you", "who are you", "how do i use this",
    }
    if any(phrase in q_lower for phrase in help_phrases_priority):
        logger.info(f"[Supervisor] Quick classify '{question}' -> HELP (priority phrase)")
        return Intent.HELP

    # Documentation/how-to questions (RAG) - for Act! CRM product help
    # Must have: (Act! keyword) OR (docs_phrase + product_action)
    act_keywords = {"act!", "act crm", "act software", "in act"}
    docs_phrases = {
        "how do i", "how to", "how can i", "where is the", "where do i find",
        "tutorial", "guide me", "documentation", "help with",
    }
    product_actions = {
        "import contacts", "export contacts", "create a group", "create group",
        "keyboard shortcut", "shortcut key", "mail merge", "marketing automation",
        "activity series", "set up", "configure",
    }

    has_act_keyword = any(kw in q_lower for kw in act_keywords)
    has_docs_phrase = any(phrase in q_lower for phrase in docs_phrases)
    has_product_action = any(action in q_lower for action in product_actions)

    # Route to DOCS if: (Act! keyword) OR (docs phrase + product action)
    # Simple feature names alone (like "opportunity") go to DATA_QUERY not DOCS
    if has_act_keyword or (has_docs_phrase and has_product_action):
        # Additional check: not asking about system capabilities
        if not any(phrase in q_lower for phrase in {"this work", "use this", "can you do"}):
            logger.info(f"[Supervisor] Quick classify '{question}' -> DOCS")
            return Intent.DOCS

    # Also route to DOCS for standalone docs phrases when not asking about data
    if has_docs_phrase:
        data_words = {"revenue", "deals", "pipeline", "sales", "contacts at", "companies", "total", "count", "stages"}
        if not any(word in q_lower for word in data_words):
            logger.info(f"[Supervisor] Quick classify '{question}' -> DOCS")
            return Intent.DOCS

    # Export indicators (check early - "export deals" should be EXPORT, not DATA_QUERY)
    export_indicators = {"export", "download", "csv", "pdf", "spreadsheet", "generate report"}
    if any(indicator in q_lower for indicator in export_indicators):
        logger.info(f"[Supervisor] Quick classify '{question}' -> EXPORT")
        return Intent.EXPORT

    # Complex multi-part indicators (check BEFORE other specialized intents)
    # A query with "and" connecting DIFFERENT operations should go to Planner
    # e.g., "Show deals and compare Q1 vs Q2" - two different operations
    # NOT "Compare Acme and TechCorp" - single comparison operation
    if " and " in q_lower:
        parts = q_lower.split(" and ")
        # Different operation indicators (not just data terms)
        operation_indicators = {"show", "list", "compare", "trend", "export", "health", "get", "find"}
        # Check if both parts have different operation indicators
        if len(parts) >= 2:
            ops_in_parts = [
                [op for op in operation_indicators if op in part]
                for part in parts[:2]
            ]
            # Complex if both parts have operations AND they're different operations
            if ops_in_parts[0] and ops_in_parts[1] and ops_in_parts[0] != ops_in_parts[1]:
                logger.info(f"[Supervisor] Quick classify '{question}' -> COMPLEX (multi-part)")
                return Intent.COMPLEX

    # Comparison indicators
    compare_indicators = {" vs ", " versus ", "compare", "comparison", "difference between"}
    if any(indicator in q_lower for indicator in compare_indicators):
        logger.info(f"[Supervisor] Quick classify '{question}' -> COMPARE")
        return Intent.COMPARE

    # Trend indicators
    trend_indicators = {
        "trend", "over time", "growth", "trajectory", "progression",
        "month over month", "year over year", "yoy", "mom",
        "by month", "by quarter", "monthly", "quarterly",
    }
    if any(indicator in q_lower for indicator in trend_indicators):
        logger.info(f"[Supervisor] Quick classify '{question}' -> TREND")
        return Intent.TREND

    # Health score indicators
    health_indicators = {"health score", "health", "healthy", "at risk", "needs attention", "account health"}
    if any(indicator in q_lower for indicator in health_indicators):
        logger.info(f"[Supervisor] Quick classify '{question}' -> HEALTH")
        return Intent.HEALTH

    # Graph relationship indicators (multi-hop entity queries)
    graph_indicators = {
        "connected to", "related to", "relationship between", "linked to",
        "path between", "network of", "chain for",
        "contacts at companies", "who works with",
        "deals associated with",
    }
    # Multi-hop: mentions 2+ entity types with relationship language
    entity_types = {"contacts", "companies", "deals", "opportunities", "activities"}
    entity_count = sum(1 for e in entity_types if e in q_lower)
    has_graph_indicator = any(indicator in q_lower for indicator in graph_indicators)
    if has_graph_indicator or (entity_count >= 2 and any(
        w in q_lower for w in {"at", "with", "whose", "where", "that have"}
    )):
        logger.info(f"[Supervisor] Quick classify '{question}' -> GRAPH")
        return Intent.GRAPH

    # Data query indicators (general catch-all for CRM data queries)
    data_indicators = {
        "show", "list", "find", "get", "what", "who", "how many", "how much",
        "revenue", "sales", "deals", "contacts", "companies", "opportunities",
        "pipeline", "renewals", "activities", "total", "average", "count"
    }
    if any(indicator in q_lower for indicator in data_indicators):
        logger.info(f"[Supervisor] Quick classify '{question}' -> DATA_QUERY")
        return Intent.DATA_QUERY

    # Help keywords - only if no data indicators matched
    help_phrases = {"help", "hello", "hi"}
    if any(phrase in q_lower for phrase in help_phrases):
        logger.info(f"[Supervisor] Quick classify '{question}' -> HELP")
        return Intent.HELP

    # Fall back to LLM for ambiguous cases
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=10)
        prompt = _CLASSIFIER_PROMPT.format(
            history=conversation_history or "No previous conversation",
            question=question,
        )

        response = llm.invoke(prompt)
        result = response.content.strip().upper()

        # Parse response
        if "DOCS" in result:
            intent = Intent.DOCS
        elif "COMPARE" in result:
            intent = Intent.COMPARE
        elif "TREND" in result:
            intent = Intent.TREND
        elif "COMPLEX" in result:
            intent = Intent.COMPLEX
        elif "EXPORT" in result:
            intent = Intent.EXPORT
        elif "HEALTH" in result:
            intent = Intent.HEALTH
        elif "GRAPH" in result:
            intent = Intent.GRAPH
        elif "DATA_QUERY" in result:
            intent = Intent.DATA_QUERY
        elif "CLARIFY" in result:
            intent = Intent.CLARIFY
        elif "HELP" in result:
            intent = Intent.HELP
        else:
            # Default to data query if unclear
            logger.warning(f"[Supervisor] Unexpected LLM response: {result}, defaulting to DATA_QUERY")
            intent = Intent.DATA_QUERY

        logger.info(f"[Supervisor] LLM classify '{question}' -> {intent.value}")
        return intent

    except Exception as e:
        logger.error(f"[Supervisor] Classification failed: {e}, defaulting to DATA_QUERY")
        return Intent.DATA_QUERY


__all__ = ["Intent", "classify_intent"]
