"""
LLM-based router for intelligent question routing.

Uses an LLM to:
1. Determine question mode (docs, data, data+docs)
2. Understand query intent
3. Extract parameters (company, timeframe, etc.)
4. Expand/rewrite queries for better understanding

Uses LangChain's .with_structured_output() for reliable Pydantic parsing.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.agent.config import get_config, is_mock_mode
from backend.agent.schemas import RouterResult
from backend.agent.datastore import get_datastore, CRMDataStore


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Role-Based Starter Detection
# =============================================================================

# Maps starter question patterns to owner IDs
# Sales Rep = jsmith, CSM = amartin, Manager = None (sees all)
STARTER_OWNER_MAP = {
    # Sales Rep starters
    "how's my pipeline": "jsmith",
    "hows my pipeline": "jsmith",
    "how is my pipeline": "jsmith",
    "show my pipeline": "jsmith",
    "what's in my pipeline": "jsmith",
    # CSM starters
    "any renewals at risk": "amartin",
    "renewals at risk": "amartin",
    "which renewals are at risk": "amartin",
    "at-risk renewals": "amartin",
    "at risk renewals": "amartin",
    # Manager starters (no owner filter - sees all)
    "how's the team doing": None,
    "hows the team doing": None,
    "how is the team doing": None,
    "team performance": None,
    "how's my team": None,
}


def detect_owner_from_starter(question: str) -> str | None:
    """
    Detect if question is a role-based starter and return owner.

    Args:
        question: The user's question

    Returns:
        Owner ID (e.g., "jsmith") or None if not a starter or Manager role
    """
    q = question.lower().strip().rstrip("?")

    for pattern, owner in STARTER_OWNER_MAP.items():
        if pattern in q:
            logger.debug(f"Detected starter pattern '{pattern}' → owner={owner}")
            return owner

    return None


# =============================================================================
# LLM Router Prompt Template (LangChain)
# =============================================================================

ROUTER_SYSTEM_PROMPT = """You are a routing assistant for Acme CRM, a customer relationship management system.

Your job is to analyze user questions and provide a complete understanding:

## DATA MODEL
The CRM has distinct data tables - route based on which table has the data:
- **companies**: Account metadata (name, industry, segment, region, status, plan, account_owner, renewal_date, health_flags)
- **contacts**: People who work AT a company (first_name, last_name, email, job_title, role)
- **opportunities**: Sales deals linked to a company (name, stage, value, expected_close_date)
- **activities**: Tasks/events (calls, emails, meetings) with due dates and owners
- **history**: Completed past interactions with notes

## ROUTING
1. **mode**: What data source should answer this question? MUST be exactly one of:
   - "docs": Help documentation, how-to guides, feature explanations, "how do I..." questions
   - "data": CRM database queries (contacts, companies, opportunities, activities, renewals, pipeline)
   - "data+docs": Questions that need both database info AND documentation context

   IMPORTANT: mode can ONLY be "docs", "data", or "data+docs". Never use any other value.
   When unsure, default to "data" for account/company questions, "docs" for how-to questions.

2. **intent**: The primary purpose of the question (separate from mode!)

   COMPANY-SPECIFIC INTENTS (require a company name):
   - "company_status": General status/summary of a specific company/account. Use for account metadata fields.
   - "pipeline": Opportunities/deals for a SPECIFIC company
   - "history": Past interactions for a SPECIFIC company
   - "contact_lookup": Get contacts (people) for a SPECIFIC company
   - "account_context": Deep context from notes/attachments for a company - use this for:
     * "Why is the deal stalled?" (needs notes about blockers)
     * "What concerns have they raised?" (needs meeting notes)
     * "Summarize our relationship" (needs comprehensive context)

   AGGREGATE/GLOBAL INTENTS (no specific company):
   - "renewals": Contract renewals across ALL accounts
   - "pipeline_summary": Total pipeline value, deal counts across ALL accounts
   - "deals_at_risk": At-risk deals - stalled, overdue, or flagged for attention
     * Pattern: "at risk", "stalled", "overdue", "stuck", "need attention"
     * "Any renewals at risk?" → deals_at_risk
     * "Which deals are stalled?" → deals_at_risk
   - "forecast": Pipeline forecast, projections, weighted pipeline
     * Pattern: "forecast", "projection", "expected", "what will close"
     * "What's the forecast for this quarter?" → forecast
   - "activities": Global activity search (recent calls, emails, meetings)
   - "contact_search": Search contacts by name or role (e.g., "Who is Maria Silva?", "Find decision makers")
   - "company_search": Search companies by segment/industry (e.g., "Show enterprise accounts")
   - "attachments": Document/file searches (e.g., "Find all proposals")
   - "analytics": Counts, breakdowns, distributions, aggregations
     * Pattern: "How many...", "What's the breakdown...", "What percentage...", "What's the distribution..."
     * "What's the breakdown of contact roles at Acme?" → analytics (with company_name)
     * "How many activities has Acme had this month?" → analytics (with company_name)
     * "Which activity type is most common?" → analytics (no company_name)

   DOCUMENTATION INTENT:
   - "general": How-to questions, feature explanations, help documentation
     * Pattern: "How do I...", "What is...", "Can you explain...", "How can I..."
     * These ask about HOW TO USE features, not to RETRIEVE data
     * NEVER confuse with data intents (e.g., "How do I add a contact?" is general, NOT contact_search)

3. **company_name**: If a specific company/account is mentioned, extract it EXACTLY as stated (null if none)
   - Extract the FULL name as the user wrote it (e.g., "Global Tech Solutions" not just "Global")
   - For partial names like "Show me Global's pipeline", extract "Global"
   - IMPORTANT: For pronouns like "their", "them", "they", "that company", or "it",
     look at CONVERSATION HISTORY to find the most recently mentioned company.
   - CRITICAL: For implicit references like "the deal", "the upgrade", "the renewal", "the opportunity",
     "the contact", look at CONVERSATION HISTORY to find which company/entity was being discussed.
     Example: If history discusses "Acme's opportunities" and question is "What stage is the upgrade deal in?",
     company_name = "Acme" because "the upgrade deal" refers to Acme's deal from context.

4. **days**: Relevant time period in days (default 30 if not specified)
   - "last 90 days" → 90, "this month" → 30, "this quarter" → 90, "recent" → 90

## QUERY UNDERSTANDING
5. **query_expansion**: A clearer, expanded version of the query that captures full user intent
   - If pronouns are used, expand them to the actual company name from conversation history

6. **key_entities**: Important entities mentioned (companies, contacts, products, metrics)

7. **action_type**: What the user wants to do
   - "retrieve": Get specific data
   - "summarize": High-level overview
   - "compare": Compare items
   - "analyze": Deep analysis

8. **confidence**: How confident you are in this analysis (0.0 to 1.0)

Analyze the question and provide your structured response."""


ROUTER_EXAMPLES = """
## MODE DECISION GUIDE:
- "data" = ONLY need CRM database (activities, pipeline, renewals, contacts, companies)
- "docs" = ONLY need help documentation (how-to, features, best practices)
- "data+docs" = Need BOTH data AND guidance on what it means or what to do

## CRITICAL: "How do I..." questions are ALWAYS docs + general intent!

## Example questions and responses:

### DOCUMENTATION QUESTIONS (mode=docs, intent=general)
# Pattern: "How do I...", "What is...", "Can you explain...", "How can I..."

Q: "How do I set up email notifications?"
{"mode": "docs", "intent": "general", "company_name": null, "days": 30,
 "query_expansion": "Explain how to configure email notifications in Acme CRM",
 "key_entities": ["email notifications"], "action_type": "retrieve", "confidence": 0.95}

Q: "What is the difference between leads and opportunities?"
{"mode": "docs", "intent": "general", "company_name": null, "days": 30,
 "query_expansion": "Explain the distinction between leads and opportunities in the CRM",
 "key_entities": ["leads", "opportunities"], "action_type": "retrieve", "confidence": 0.95}

Q: "Can you explain how tags work?"
{"mode": "docs", "intent": "general", "company_name": null, "days": 30,
 "query_expansion": "Explain the tagging system and how to use tags in Acme CRM",
 "key_entities": ["tags"], "action_type": "retrieve", "confidence": 0.95}

### COMPANY-SPECIFIC DATA QUERIES
Q: "What's the pipeline for Acme Corp?"
{"mode": "data", "intent": "pipeline", "company_name": "Acme Corp", "days": 30,
 "query_expansion": "Show open opportunities for Acme Corp",
 "key_entities": ["Acme Corp"], "action_type": "retrieve", "confidence": 0.95}

Q: "Show me Acme's contacts"
{"mode": "data", "intent": "contact_lookup", "company_name": "Acme", "days": 30,
 "query_expansion": "List contacts associated with Acme",
 "key_entities": ["Acme", "contacts"], "action_type": "retrieve", "confidence": 0.95}

Q: "What's the status of Beta Tech?"
{"mode": "data", "intent": "company_status", "company_name": "Beta Tech", "days": 90,
 "query_expansion": "Provide status summary for Beta Tech",
 "key_entities": ["Beta Tech"], "action_type": "summarize", "confidence": 0.95}

### GLOBAL/AGGREGATE DATA QUERIES (no specific company)
Q: "Find John Patterson"
{"mode": "data", "intent": "contact_search", "company_name": null, "days": 30,
 "query_expansion": "Search for contact named John Patterson",
 "key_entities": ["John Patterson"], "action_type": "retrieve", "confidence": 0.95}

Q: "Who are the executive sponsors in our accounts?"
{"mode": "data", "intent": "contact_search", "company_name": null, "days": 30,
 "query_expansion": "List contacts with Executive Sponsor role",
 "key_entities": ["executive sponsors"], "action_type": "retrieve", "confidence": 0.95}

Q: "List mid-market segment companies"
{"mode": "data", "intent": "company_search", "company_name": null, "days": 30,
 "query_expansion": "List companies in Mid-Market segment",
 "key_entities": ["mid-market", "companies"], "action_type": "retrieve", "confidence": 0.95}

Q: "How many open deals do we have?"
{"mode": "data", "intent": "pipeline_summary", "company_name": null, "days": 30,
 "query_expansion": "Show count of open deals across all accounts",
 "key_entities": ["deals", "open"], "action_type": "summarize", "confidence": 0.95}

Q: "Which accounts have upcoming renewals?"
{"mode": "data", "intent": "renewals", "company_name": null, "days": 90,
 "query_expansion": "List accounts with renewals in the next 90 days",
 "key_entities": ["renewals"], "action_type": "retrieve", "confidence": 0.95}

### DEALS AT RISK (stalled, overdue deals)
Q: "Any renewals at risk?"
{"mode": "data", "intent": "deals_at_risk", "company_name": null, "days": 90,
 "query_expansion": "Show deals that are at risk or stalled",
 "key_entities": ["renewals", "at-risk"], "action_type": "retrieve", "confidence": 0.95}

Q: "Which deals are stalled?"
{"mode": "data", "intent": "deals_at_risk", "company_name": null, "days": 90,
 "query_expansion": "List deals that have been stalled or stuck",
 "key_entities": ["deals", "stalled"], "action_type": "retrieve", "confidence": 0.95}

### FORECAST (pipeline projections)
Q: "What's the forecast for this quarter?"
{"mode": "data", "intent": "forecast", "company_name": null, "days": 90,
 "query_expansion": "Show weighted pipeline forecast for the quarter",
 "key_entities": ["forecast", "quarter"], "action_type": "summarize", "confidence": 0.95}

Q: "How much pipeline will close this month?"
{"mode": "data", "intent": "forecast", "company_name": null, "days": 30,
 "query_expansion": "Calculate expected pipeline closure for the month",
 "key_entities": ["pipeline", "close"], "action_type": "summarize", "confidence": 0.95}

### FORECAST ACCURACY (win rate metrics)
Q: "What's our win rate?"
{"mode": "data", "intent": "forecast_accuracy", "company_name": null, "days": 30,
 "query_expansion": "Show overall win rate from closed deals",
 "key_entities": ["win rate", "accuracy"], "action_type": "summarize", "confidence": 0.95}

Q: "How accurate are our forecasts?"
{"mode": "data", "intent": "forecast_accuracy", "company_name": null, "days": 30,
 "query_expansion": "Show forecast accuracy based on historical closed deals",
 "key_entities": ["forecast", "accuracy"], "action_type": "summarize", "confidence": 0.95}

Q: "Search for contract documents"
{"mode": "data", "intent": "attachments", "company_name": null, "days": 30,
 "query_expansion": "Find documents containing contracts",
 "key_entities": ["contracts"], "action_type": "retrieve", "confidence": 0.95}

Q: "What tasks are pending?"
{"mode": "data", "intent": "activities", "company_name": null, "days": 30,
 "query_expansion": "List pending tasks and activities",
 "key_entities": ["tasks", "pending"], "action_type": "retrieve", "confidence": 0.95}

### ACCOUNT CONTEXT (deep unstructured search)
Q: "Why is the Acme deal stalled?"
{"mode": "data", "intent": "account_context", "company_name": "Acme", "days": 90,
 "query_expansion": "Search notes for blockers on Acme deal",
 "key_entities": ["Acme", "stalled"], "action_type": "analyze", "confidence": 0.95}

### ANALYTICS QUERIES (counts, breakdowns, distributions)
# Pattern: "How many...", "What's the breakdown...", "percentage", "distribution", "count"

Q: "How many calls did we make last week?"
{"mode": "data", "intent": "analytics", "company_name": null, "days": 7,
 "query_expansion": "Count call activities in the last 7 days",
 "key_entities": ["calls"], "action_type": "summarize", "confidence": 0.95}

Q: "What percentage of contacts are decision makers?"
{"mode": "data", "intent": "analytics", "company_name": null, "days": 30,
 "query_expansion": "Calculate percentage of contacts with decision maker role",
 "key_entities": ["contacts", "decision makers"], "action_type": "summarize", "confidence": 0.95}

Q: "Show me activity counts by type"
{"mode": "data", "intent": "analytics", "company_name": null, "days": 30,
 "query_expansion": "Break down activities by type",
 "key_entities": ["activities"], "action_type": "summarize", "confidence": 0.95}

### TEAM PERFORMANCE QUESTIONS (Manager view - no owner filter)
# Pattern: "team", "team doing", "team performance", "my team"
# These show aggregate pipeline/activity metrics across all reps

Q: "Give me a team overview"
{"mode": "data", "intent": "pipeline_summary", "company_name": null, "days": 30,
 "query_expansion": "Show aggregate team pipeline performance and metrics",
 "key_entities": ["team", "pipeline"], "action_type": "summarize", "confidence": 0.95}

Q: "Show team metrics for this month"
{"mode": "data", "intent": "pipeline_summary", "company_name": null, "days": 30,
 "query_expansion": "Show team pipeline and activity metrics",
 "key_entities": ["team", "metrics"], "action_type": "summarize", "confidence": 0.95}

Q: "What's the overall team status?"
{"mode": "data", "intent": "pipeline_summary", "company_name": null, "days": 30,
 "query_expansion": "Show my team's aggregate pipeline and activity metrics",
 "key_entities": ["team", "status"], "action_type": "summarize", "confidence": 0.95}

### COMBINED DATA + DOCS
Q: "Which accounts are at risk and what should I do?"
{"mode": "data+docs", "intent": "renewals", "company_name": null, "days": 90,
 "query_expansion": "Identify at-risk accounts and provide churn prevention guidance",
 "key_entities": ["at-risk"], "action_type": "analyze", "confidence": 0.9}

### COMPANY-SPECIFIC ACTIVITIES (use company_status, NOT activities)
Q: "Show me recent activities for Skyline Industries"
{"mode": "data", "intent": "company_status", "company_name": "Skyline Industries", "days": 90,
 "query_expansion": "Show recent activities for Skyline Industries",
 "key_entities": ["Skyline Industries", "activities"], "action_type": "retrieve", "confidence": 0.95}

### PRONOUN RESOLUTION (requires conversation history)
# Given conversation history: "User asked about Northwind Corp"
Q: "What about their contacts?"
{"mode": "data", "intent": "contact_lookup", "company_name": "Northwind Corp", "days": 30,
 "query_expansion": "List contacts for Northwind Corp",
 "key_entities": ["Northwind Corp", "contacts"], "action_type": "retrieve", "confidence": 0.9}

### IMPLICIT CONTEXT RESOLUTION (requires conversation history)
# Given conversation history: "User: Show me Acme Manufacturing's opportunities / Assistant: [listed opportunities including upgrade deal]"
Q: "What stage is the upgrade deal in?"
{"mode": "data", "intent": "pipeline", "company_name": "Acme Manufacturing", "days": 30,
 "query_expansion": "What stage is Acme Manufacturing's upgrade deal opportunity in?",
 "key_entities": ["Acme Manufacturing", "upgrade deal"], "action_type": "retrieve", "confidence": 0.9}

# Given conversation history: "User asked about Beta Tech's pipeline"
Q: "What's the deal worth?"
{"mode": "data", "intent": "pipeline", "company_name": "Beta Tech", "days": 30,
 "query_expansion": "What is the value of Beta Tech's deal?",
 "key_entities": ["Beta Tech", "deal value"], "action_type": "retrieve", "confidence": 0.9}
"""

# =============================================================================
# LangChain Prompt Template
# =============================================================================

ROUTER_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human", """{examples}

{conversation_context}Now analyze this question:
Q: "{question}"
"""),
])


# =============================================================================
# LLM Router Implementation
# =============================================================================

class LLMRouterError(Exception):
    """Custom exception for LLM router failures."""
    pass


class LLMRouterResponse(BaseModel):
    """Pydantic model for structured LLM router output.

    Used with .with_structured_output() for reliable parsing.
    """
    mode: Literal["docs", "data", "data+docs"] = Field(
        default="data+docs",
        description="The data source mode: 'docs' for documentation, 'data' for CRM data, 'data+docs' for both"
    )
    intent: Literal[
        "company_status",  # General status/summary of a specific company
        "renewals",        # Contract renewals, expirations
        "pipeline",        # Company-specific opportunities/deals
        "pipeline_summary", # Aggregate pipeline across all accounts
        "deals_at_risk",   # At-risk deals (stalled, overdue)
        "forecast",        # Pipeline forecast/projections
        "forecast_accuracy", # Win rate and accuracy metrics
        "activities",      # Global activity search (calls, emails, meetings)
        "history",         # Past interactions for a company
        "account_context", # Deep context from notes/attachments
        "contact_lookup",  # Contacts for a specific company
        "contact_search",  # Global contact search by name/role
        "company_search",  # Search companies by criteria (segment, industry)
        "attachments",     # Document/file searches
        "analytics",       # Counts, breakdowns, aggregations (how many, breakdown, distribution)
        "general",         # Documentation/help questions
    ] = Field(
        default="general",
        description="The primary intent of the question"
    )
    company_name: str | None = Field(
        default=None,
        description="The company/account name mentioned in the question, if any"
    )
    days: int = Field(
        default=30, ge=1, le=365,
        description="Time period in days (e.g., 'last 90 days' -> 90)"
    )
    query_expansion: str | None = Field(
        default=None,
        description="A clearer, expanded version of the query"
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Important entities mentioned (companies, contacts, products)"
    )
    action_type: Literal["retrieve", "summarize", "compare", "analyze"] = Field(
        default="retrieve",
        description="What the user wants to do with the information"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence in this analysis (0.0 to 1.0)"
    )


# Cached structured LLM chain
_router_chain = None


def clear_router_chain_cache():
    """Clear the cached router chain. Call when prompts change."""
    global _router_chain
    _router_chain = None
    logger.debug("Router chain cache cleared")


def _get_router_chain():
    """Get or create the cached router chain with structured output."""
    global _router_chain
    if _router_chain is not None:
        return _router_chain

    import os
    config = get_config()

    llm = ChatOpenAI(
        model=config.router_model,
        temperature=config.router_temperature,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_retries=3,
    )

    # Use structured output for reliable Pydantic parsing
    structured_llm = llm.with_structured_output(LLMRouterResponse)

    # Create LCEL chain: prompt | structured_llm
    _router_chain = ROUTER_PROMPT_TEMPLATE | structured_llm

    logger.debug(f"Created router chain with model={config.router_model}")
    return _router_chain


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
def _call_llm_router(question: str, conversation_history: str = "") -> dict:
    """Call LLM for routing using LCEL chain with structured output."""
    logger.debug(f"LLM Router: Analyzing question: {question[:50]}...")

    # Build conversation context section
    conversation_context = ""
    if conversation_history:
        conversation_context = f"CONVERSATION HISTORY:\n{conversation_history}\n\n"

    # Get the cached chain
    chain = _get_router_chain()

    # Invoke the chain with structured output
    result: LLMRouterResponse = chain.invoke({
        "examples": ROUTER_EXAMPLES,
        "conversation_context": conversation_context,
        "question": question,
    })

    logger.debug(f"LLM Router: Structured output received: mode={result.mode}, intent={result.intent}")

    return result.model_dump()


def llm_route_question(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    datastore: CRMDataStore | None = None,
    conversation_history: str = "",
) -> RouterResult:
    """
    Route a question using LLM intelligence.

    Args:
        question: The user's question
        mode: Explicit mode override ("auto" for LLM decision)
        company_id: Pre-specified company ID
        datastore: Optional datastore instance
        conversation_history: Formatted conversation history for context

    Returns:
        RouterResult with routing decision and extracted parameters
    """
    ds = datastore or get_datastore()

    # If mode is explicitly set, return minimal routing
    if mode and mode != "auto":
        logger.debug(f"Mode explicitly set to '{mode}'")
        return RouterResult(
            mode_used=mode,
            company_id=company_id,
            days=30,
            intent="general",
            owner=detect_owner_from_starter(question),
        )

    # Mock mode returns default routing (for testing without API)
    if is_mock_mode():
        logger.debug("Mock mode: returning default routing")
        return RouterResult(
            mode_used="data+docs",
            company_id=company_id,
            days=30,
            intent="general",
            owner=detect_owner_from_starter(question),
        )

    # LLM routing
    llm_result = _call_llm_router(question, conversation_history)

    logger.info(
        f"LLM Router: mode={llm_result['mode']}, "
        f"intent={llm_result['intent']}, "
        f"confidence={llm_result.get('confidence', 'N/A')}"
    )

    # Resolve company ID if LLM found a company name
    resolved_company = company_id
    if not resolved_company and llm_result.get("company_name"):
        resolved_company = ds.resolve_company_id(llm_result["company_name"])
        if resolved_company:
            logger.debug(f"Resolved company '{llm_result['company_name']}' to ID: {resolved_company}")

    # Detect owner from starter pattern (role-based filtering)
    owner = detect_owner_from_starter(question)

    return RouterResult(
        mode_used=llm_result["mode"],
        company_id=resolved_company,
        days=llm_result.get("days", 30),
        intent=llm_result.get("intent", "general"),
        query_expansion=llm_result.get("query_expansion"),
        llm_confidence=llm_result.get("confidence"),
        key_entities=llm_result.get("key_entities", []),
        action_type=llm_result.get("action_type"),
        owner=owner,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def route_question(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    datastore: CRMDataStore | None = None,
    conversation_history: str = "",
) -> RouterResult:
    """
    Main routing function - uses LLM for intelligent routing.

    This is the recommended entry point for routing questions.
    Merges routing + query understanding into a single LLM call.

    Args:
        question: The user's question
        mode: Explicit mode override ("auto" for LLM decision)
        company_id: Pre-specified company ID
        datastore: Optional datastore instance
        conversation_history: Formatted conversation history for context
    """
    result = llm_route_question(question, mode, company_id, datastore, conversation_history)

    # Ensure owner is detected for role-based starters
    if result.owner is None:
        result.owner = detect_owner_from_starter(question)

    return result


