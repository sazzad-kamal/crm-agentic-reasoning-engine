"""
LLM-based router for intelligent question routing.

Uses an LLM to:
1. Determine question mode (docs, data, data+docs)
2. Understand query intent
3. Extract parameters (company, timeframe, etc.)
4. Expand/rewrite queries for better understanding

Falls back to heuristic router on LLM failures.
"""

import logging
import re
from typing import Optional, Literal

from pydantic import BaseModel, Field
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type,
)

from backend.agent.config import get_config, is_mock_mode
from backend.agent.schemas import RouterResult
from backend.agent.datastore import get_datastore, CRMDataStore
from backend.agent import router as heuristic_router  # Fallback
from backend.common.llm_client import call_llm


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# LLM Router Prompts
# =============================================================================

ROUTER_SYSTEM_PROMPT = """You are a routing assistant for Acme CRM, a customer relationship management system.

Your job is to analyze user questions and provide a complete understanding:

## ROUTING
1. **mode**: What data source should answer this question?
   - "docs": Help documentation, how-to guides, feature explanations
   - "data": CRM database queries (contacts, companies, opportunities, activities)
   - "data+docs": Questions that need both database info AND documentation context

2. **intent**: The primary purpose of the question
   - "company_status": General status/summary of a company/account
   - "renewals": Questions about contract renewals, expirations
   - "pipeline": Sales pipeline, opportunities, deals
   - "activities": Calls, emails, meetings, tasks
   - "history": Past interactions, what happened previously
   - "general": General questions or unclear intent

3. **company_name**: If a specific company/account is mentioned, extract it (null if none)

4. **days**: Relevant time period in days (default 30 if not specified)
   - "last 90 days" → 90, "this month" → 30, "this quarter" → 90, "recent" → 90

## QUERY UNDERSTANDING
5. **query_expansion**: A clearer, expanded version of the query that captures full user intent

6. **key_entities**: Important entities mentioned (companies, contacts, products, metrics)

7. **action_type**: What the user wants to do
   - "retrieve": Get specific data
   - "summarize": High-level overview
   - "compare": Compare items
   - "analyze": Deep analysis

8. **confidence**: How confident you are in this analysis (0.0 to 1.0)

Respond ONLY with valid JSON in this exact format:
{
    "mode": "docs" | "data" | "data+docs",
    "intent": "company_status" | "renewals" | "pipeline" | "activities" | "history" | "general",
    "company_name": "Company Name" | null,
    "days": 30,
    "query_expansion": "Expanded query text",
    "key_entities": ["entity1", "entity2"],
    "action_type": "retrieve" | "summarize" | "compare" | "analyze",
    "confidence": 0.0 to 1.0
}"""


ROUTER_EXAMPLES = """
Example questions and responses:

Q: "What's going on with Acme Manufacturing in the last 90 days?"
{
    "mode": "data+docs",
    "intent": "company_status",
    "company_name": "Acme Manufacturing",
    "days": 90,
    "query_expansion": "Provide a comprehensive status summary for Acme Manufacturing including recent activities, open opportunities, any upcoming renewals, and key updates from the last 90 days",
    "key_entities": ["Acme Manufacturing"],
    "action_type": "summarize",
    "confidence": 0.95
}

Q: "How do I create a new opportunity?"
{
    "mode": "docs",
    "intent": "general",
    "company_name": null,
    "days": 30,
    "query_expansion": "Explain the steps and process for creating a new sales opportunity in Acme CRM, including any required fields and best practices",
    "key_entities": ["opportunity"],
    "action_type": "retrieve",
    "confidence": 0.9
}

Q: "Which accounts have upcoming renewals in the next 90 days?"
{
    "mode": "data",
    "intent": "renewals",
    "company_name": null,
    "days": 90,
    "query_expansion": "List all accounts with contract renewals due within the next 90 days, including renewal dates and contract values",
    "key_entities": ["renewals", "accounts"],
    "action_type": "retrieve",
    "confidence": 0.95
}

Q: "Compare pipeline values between Q3 and Q4"
{
    "mode": "data",
    "intent": "pipeline",
    "company_name": null,
    "days": 180,
    "query_expansion": "Compare the total pipeline value, deal count, and stage distribution between Q3 and Q4 quarters",
    "key_entities": ["pipeline", "Q3", "Q4"],
    "action_type": "compare",
    "confidence": 0.85
}
"""


# =============================================================================
# LLM Router Implementation  
# =============================================================================

class LLMRouterError(Exception):
    """Custom exception for LLM router failures."""
    pass


class LLMRouterResponse(BaseModel):
    """Pydantic model for parsing LLM router JSON responses."""
    mode: Literal["docs", "data", "data+docs"] = "data+docs"
    intent: Literal["company_status", "renewals", "pipeline", "activities", "history", "general"] = "general"
    company_name: str | None = None
    days: int = Field(default=30, ge=1, le=365)
    query_expansion: str | None = None
    key_entities: list[str] = Field(default_factory=list)
    action_type: Literal["retrieve", "summarize", "compare", "analyze"] = "retrieve"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @classmethod
    def from_llm_text(cls, text: str) -> "LLMRouterResponse":
        """Parse LLM response text, extracting JSON from markdown if needed."""
        text = text.strip()
        # Extract JSON from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return cls.model_validate_json(text)


def _parse_router_response(response_text: str) -> dict:
    """Parse and validate LLM router response using Pydantic."""
    try:
        parsed = LLMRouterResponse.from_llm_text(response_text)
        return parsed.model_dump()
    except Exception as e:
        raise LLMRouterError(f"Failed to parse router response: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
def _call_llm_router(question: str) -> dict:
    """Call LLM for routing with retry logic."""
    config = get_config()
    
    prompt = f"{ROUTER_EXAMPLES}\n\nNow analyze this question:\nQ: \"{question}\""
    
    logger.debug(f"LLM Router: Analyzing question: {question[:50]}...")
    
    response = call_llm(
        prompt=prompt,
        system_prompt=ROUTER_SYSTEM_PROMPT,
        model=config.router_model,
        temperature=config.router_temperature,
        max_tokens=256,
    )
    
    return _parse_router_response(response)


def llm_route_question(
    question: str,
    mode: str = "auto",
    company_id: Optional[str] = None,
    datastore: Optional[CRMDataStore] = None,
) -> RouterResult:
    """
    Route a question using LLM intelligence.
    
    Falls back to heuristic router on failure.
    
    Args:
        question: The user's question
        mode: Explicit mode override ("auto" for LLM decision)
        company_id: Pre-specified company ID
        datastore: Optional datastore instance
        
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
            llm_result = _call_llm_router(question)
            
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
    company_id: Optional[str] = None,
    datastore: Optional[CRMDataStore] = None,
) -> RouterResult:
    """
    Main routing function - uses LLM or heuristics based on config.
    
    This is the recommended entry point for routing questions.
    Merges routing + query understanding into a single LLM call.
    """
    config = get_config()
    
    if config.use_llm_router:
        return llm_route_question(question, mode, company_id, datastore)
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
