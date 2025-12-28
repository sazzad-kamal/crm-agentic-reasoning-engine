"""
LLM-based router for intelligent question routing.

Uses an LLM to:
1. Determine question mode (docs, data, data+docs)
2. Understand query intent
3. Extract parameters (company, timeframe, etc.)
4. Expand/rewrite queries for better understanding

Falls back to heuristic router on LLM failures.

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
from backend.agent import router as heuristic_router  # Fallback


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# LLM Router Prompt Template (LangChain)
# =============================================================================

ROUTER_SYSTEM_PROMPT = """You are a routing assistant for Acme CRM, a customer relationship management system.

Your job is to analyze user questions and provide a complete understanding:

## ROUTING
1. **mode**: What data source should answer this question? MUST be exactly one of:
   - "docs": Help documentation, how-to guides, feature explanations, "how do I..." questions
   - "data": CRM database queries (contacts, companies, opportunities, activities, renewals, pipeline)
   - "data+docs": Questions that need both database info AND documentation context

   IMPORTANT: mode can ONLY be "docs", "data", or "data+docs". Never use any other value.
   When unsure, default to "data" for account/company questions, "docs" for how-to questions.

2. **intent**: The primary purpose of the question (separate from mode!)

   COMPANY-SPECIFIC INTENTS (require a company name):
   - "company_status": General status/summary of a specific company/account
   - "pipeline": Opportunities/deals for a SPECIFIC company
   - "history": Past interactions for a SPECIFIC company
   - "contact_lookup": Get contacts for a SPECIFIC company (e.g., "show me Acme's contacts")
   - "account_context": Deep context from notes/attachments for a company - use this for:
     * "Why is the deal stalled?" (needs notes about blockers)
     * "What concerns have they raised?" (needs meeting notes)
     * "Summarize our relationship" (needs comprehensive context)

   AGGREGATE/GLOBAL INTENTS (no specific company):
   - "renewals": Contract renewals across ALL accounts
   - "pipeline_summary": Total pipeline value, deal counts across ALL accounts
   - "activities": Global activity search (recent calls, emails, meetings)
   - "contact_search": Search contacts by name or role (e.g., "Who is Maria Silva?", "Find decision makers")
   - "company_search": Search companies by segment/industry (e.g., "Show enterprise accounts")
   - "groups": Account group queries (e.g., "Who is in the at-risk group?")
   - "attachments": Document/file searches (e.g., "Find all proposals")

   DOCUMENTATION INTENT:
   - "general": How-to questions, feature explanations, help documentation
     * "How do I create a contact?" → general (NOT contact_search!)
     * "What are pipeline stages?" → general
     * "How does email marketing work?" → general

3. **company_name**: If a specific company/account is mentioned, extract it (null if none)
   - IMPORTANT: If the user uses pronouns like "their", "them", "they", "that company", or "it",
     look at the CONVERSATION HISTORY to find the most recently mentioned company and use that name.

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
Q: "How do I create a new contact?"
{"mode": "docs", "intent": "general", "company_name": null, "days": 30,
 "query_expansion": "Explain how to create a new contact in Acme CRM",
 "key_entities": ["contact"], "action_type": "retrieve", "confidence": 0.95}

Q: "What are the pipeline stages?"
{"mode": "docs", "intent": "general", "company_name": null, "days": 30,
 "query_expansion": "Explain the different pipeline stages in Acme CRM",
 "key_entities": ["pipeline stages"], "action_type": "retrieve", "confidence": 0.95}

Q: "How does email marketing work?"
{"mode": "docs", "intent": "general", "company_name": null, "days": 30,
 "query_expansion": "Explain the email marketing campaign feature in Acme CRM",
 "key_entities": ["email marketing"], "action_type": "retrieve", "confidence": 0.95}

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
Q: "Who is Maria Silva?"
{"mode": "data", "intent": "contact_search", "company_name": null, "days": 30,
 "query_expansion": "Find contact named Maria Silva",
 "key_entities": ["Maria Silva"], "action_type": "retrieve", "confidence": 0.95}

Q: "Find all decision makers"
{"mode": "data", "intent": "contact_search", "company_name": null, "days": 30,
 "query_expansion": "List contacts with Decision Maker role",
 "key_entities": ["decision makers"], "action_type": "retrieve", "confidence": 0.95}

Q: "Show me all enterprise accounts"
{"mode": "data", "intent": "company_search", "company_name": null, "days": 30,
 "query_expansion": "List companies in Enterprise segment",
 "key_entities": ["enterprise", "accounts"], "action_type": "retrieve", "confidence": 0.95}

Q: "What's the total pipeline value?"
{"mode": "data", "intent": "pipeline_summary", "company_name": null, "days": 30,
 "query_expansion": "Show aggregate pipeline value across all accounts",
 "key_entities": ["pipeline", "total"], "action_type": "summarize", "confidence": 0.95}

Q: "Which accounts have upcoming renewals?"
{"mode": "data", "intent": "renewals", "company_name": null, "days": 90,
 "query_expansion": "List accounts with renewals in the next 90 days",
 "key_entities": ["renewals"], "action_type": "retrieve", "confidence": 0.95}

Q: "Who is in the at-risk accounts group?"
{"mode": "data", "intent": "groups", "company_name": null, "days": 30,
 "query_expansion": "List members of the at-risk accounts group",
 "key_entities": ["at-risk", "group"], "action_type": "retrieve", "confidence": 0.95}

Q: "Find all proposals"
{"mode": "data", "intent": "attachments", "company_name": null, "days": 30,
 "query_expansion": "Search for proposal documents",
 "key_entities": ["proposals"], "action_type": "retrieve", "confidence": 0.95}

Q: "Show recent activities"
{"mode": "data", "intent": "activities", "company_name": null, "days": 30,
 "query_expansion": "List recent calls, emails, and meetings",
 "key_entities": ["activities"], "action_type": "retrieve", "confidence": 0.95}

### ACCOUNT CONTEXT (deep unstructured search)
Q: "Why is the Acme deal stalled?"
{"mode": "data", "intent": "account_context", "company_name": "Acme", "days": 90,
 "query_expansion": "Search notes for blockers on Acme deal",
 "key_entities": ["Acme", "stalled"], "action_type": "analyze", "confidence": 0.95}

### COMBINED DATA + DOCS
Q: "Which accounts are at risk and what should I do?"
{"mode": "data+docs", "intent": "renewals", "company_name": null, "days": 90,
 "query_expansion": "Identify at-risk accounts and provide churn prevention guidance",
 "key_entities": ["at-risk"], "action_type": "analyze", "confidence": 0.9}
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
        "activities",      # Global activity search (calls, emails, meetings)
        "history",         # Past interactions for a company
        "account_context", # Deep context from notes/attachments
        "contact_lookup",  # Contacts for a specific company
        "contact_search",  # Global contact search by name/role
        "company_search",  # Search companies by criteria (segment, industry)
        "groups",          # Account groups queries
        "attachments",     # Document/file searches
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

    Falls back to heuristic router on failure.

    Args:
        question: The user's question
        mode: Explicit mode override ("auto" for LLM decision)
        company_id: Pre-specified company ID
        datastore: Optional datastore instance
        conversation_history: Formatted conversation history for context

    Returns:
        RouterResult with routing decision and extracted parameters
    """
    config = get_config()
    ds = datastore or get_datastore()

    # If mode is explicitly set, skip LLM routing
    if mode and mode != "auto":
        logger.debug(f"Mode explicitly set to '{mode}', skipping LLM router")
        return heuristic_router.route_question(question, mode, company_id, datastore)

    # Skip LLM in mock mode
    if is_mock_mode():
        logger.debug("Mock mode: Using heuristic router")
        return heuristic_router.route_question(question, mode, company_id, datastore)

    # Try LLM routing
    if config.use_llm_router:
        try:
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
            
            return RouterResult(
                mode_used=llm_result["mode"],
                company_id=resolved_company,
                days=llm_result.get("days", 30),
                intent=llm_result.get("intent", "general"),
                query_expansion=llm_result.get("query_expansion"),
                llm_confidence=llm_result.get("confidence"),
                key_entities=llm_result.get("key_entities", []),
                action_type=llm_result.get("action_type"),
            )
            
        except LLMRouterError as e:
            logger.warning(f"LLM router response error: {e}")
            if config.fallback_to_heuristics:
                logger.info("Falling back to heuristic router")
            else:
                raise
        except Exception as e:
            logger.warning(f"LLM router failed: {e}")
            if config.fallback_to_heuristics:
                logger.info("Falling back to heuristic router")
            else:
                raise
    
    # Fallback to heuristic router
    return heuristic_router.route_question(question, mode, company_id, datastore)


# =============================================================================
# Unified Router Function
# =============================================================================

def route_question(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    datastore: CRMDataStore | None = None,
    conversation_history: str = "",
) -> RouterResult:
    """
    Main routing function - uses LLM or heuristics based on config.

    This is the recommended entry point for routing questions.
    Merges routing + query understanding into a single LLM call.

    Args:
        question: The user's question
        mode: Explicit mode override ("auto" for LLM decision)
        company_id: Pre-specified company ID
        datastore: Optional datastore instance
        conversation_history: Formatted conversation history for context
    """
    config = get_config()

    if config.use_llm_router:
        return llm_route_question(question, mode, company_id, datastore, conversation_history)
    else:
        return heuristic_router.route_question(question, mode, company_id, datastore)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing LLM Router")
    print("=" * 60)
    
    test_questions = [
        "What's going on with Acme Manufacturing in the last 90 days?",
        "How do I create a new opportunity?",
        "Which accounts have upcoming renewals in the next 90 days?",
        "Show the open pipeline for Beta Tech Solutions",
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        result = route_question(q)
        print(f"   Mode: {result.mode_used}")
        print(f"   Company: {result.company_id}")
        print(f"   Days: {result.days}")
        print(f"   Intent: {result.intent}")
        print(f"   Query Expansion: {result.query_expansion}")
