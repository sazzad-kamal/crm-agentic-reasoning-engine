"""
Router for determining question mode and extracting parameters.

Uses heuristics to decide between:
- "docs": Help documentation questions
- "data": CRM data questions  
- "data+docs": Combined (default for ambiguous cases)
"""

import re
from backend.agent.schemas import RouterResult
from backend.agent.datastore import get_datastore, CRMDataStore


# =============================================================================
# Heuristic Patterns
# =============================================================================

# Patterns suggesting documentation questions
DOCS_PATTERNS = [
    r"\bhow\s+(do|can|to)\b",
    r"\bwhat\s+is\b",
    r"\bwhere\s+(can|do)\b",
    r"\bexplain\b",
    r"\bdocument(ation)?\b",
    r"\bhelp\b",
    r"\bguide\b",
    r"\btutorial\b",
    r"\bsteps?\s+to\b",
    r"\bhow\s+does\b",
    r"\bwhat\s+are\s+the\s+(steps|ways|options)\b",
    r"\blearn(ing)?\s+about\b",
    r"\bfeatures?\b",
    r"\bconfigur(e|ation)\b",
    r"\bsettings?\b",
    r"\bset\s*up\b",
]

# Patterns suggesting CRM data questions
DATA_PATTERNS = [
    r"\blast\s+\d+\s+days?\b",
    r"\brecent(ly)?\b",
    r"\bactivit(y|ies)\b",
    r"\bpipeline\b",
    r"\bopportunit(y|ies)\b",
    r"\brenewal(s)?\b",
    r"\bstatus\b",
    r"\bgoing\s+on\b",
    r"\bhappening\b",
    r"\bupdate(s)?\b",
    r"\bhistory\b",
    r"\bcall(s)?\b",
    r"\bemail(s)?\b",
    r"\bmeeting(s)?\b",
    r"\bdeal(s)?\b",
    r"\bcontact(s)?\b",
    r"\baccount(s)?\b",
    r"\bcompan(y|ies)\b",
    r"\bopen\s+deals?\b",
    r"\bclosed?\b",
    r"\bwon\b",
    r"\blost\b",
    r"\bstage\b",
    r"\bvalue\b",
    r"\brevenue\b",
    r"\bupcoming\b",
    r"\bdue\b",
    r"\boverdue\b",
    r"\bshow\s+(me\s+)?the\b",
    r"\blist\b",
    r"\bget\b",
    r"\bfind\b",
]

# Patterns to extract timeframe
TIMEFRAME_PATTERNS = [
    (r"\blast\s+(\d+)\s+days?\b", lambda m: int(m.group(1))),
    (r"\bpast\s+(\d+)\s+days?\b", lambda m: int(m.group(1))),
    (r"\bnext\s+(\d+)\s+days?\b", lambda m: int(m.group(1))),
    (r"\b(\d+)\s+days?\s+ago\b", lambda m: int(m.group(1))),
    (r"\bthis\s+month\b", lambda m: 30),
    (r"\bthis\s+quarter\b", lambda m: 90),
    (r"\bthis\s+year\b", lambda m: 365),
    (r"\blast\s+month\b", lambda m: 30),
    (r"\blast\s+quarter\b", lambda m: 90),
    (r"\blast\s+year\b", lambda m: 365),
    (r"\brecent(ly)?\b", lambda m: 90),
]

# Intent patterns
INTENT_PATTERNS = {
    "company_status": [
        r"\bwhat['']?s\s+(going\s+on|happening)\b",
        r"\bstatus\s+(of|for|with)\b",
        r"\bupdate\s+(on|for|about)\b",
        r"\bhow\s+is\b.*\bdoing\b",
        r"\bsummar(y|ize)\b",
    ],
    "renewals": [
        r"\brenewal(s)?\b",
        r"\brenew(ing)?\b",
        r"\bexpir(e|ing|ation)\b",
        r"\bupcoming\s+renewal\b",
        r"\bdue\s+for\s+renewal\b",
    ],
    "pipeline": [
        r"\bpipeline\b",
        r"\bopen\s+(deals?|opportunities?)\b",
        r"\bopportunit(y|ies)\b",
        r"\bsales\s+pipeline\b",
        r"\bdeals?\s+in\s+progress\b",
    ],
    "pipeline_summary": [
        r"\b(total|all|overall|aggregate)\s+(pipeline|deals?|opportunities?)\b",
        r"\bpipeline\s+(overview|summary|total)\b",
        r"\bhow\s+(many|much)\b.*\b(deals?|opportunities?|pipeline)\b",
        r"\btotal\s+(value|amount)\b",
        r"\bacross\s+all\b",
    ],
    "deals_at_risk": [
        r"\bat\s*risk\b",
        r"\bstalled\b",
        r"\bstuck\b",
        r"\boverdue\b",
        r"\bneeds?\s+attention\b",
        r"\bblocked\b",
        r"\bat-risk\b",
    ],
    "forecast": [
        r"\bforecast\b",
        r"\bprojection(s)?\b",
        r"\bexpected\s+(revenue|close|value)\b",
        r"\bwhat\s+will\s+close\b",
        r"\bweighted\s+(pipeline|value)\b",
        r"\b(forecast|revenue)\s+(summary|total|across)\b",
        r"\bforecast\s+(this|next|for)\b",
    ],
    "activities": [
        r"\bactivit(y|ies)\b",
        r"\bcall(s)?\b",
        r"\bemail(s)?\b",
        r"\bmeeting(s)?\b",
        r"\btask(s)?\b",
        r"\btouch\s*point(s)?\b",
    ],
    "history": [
        r"\bhistory\b",
        r"\bprevious\b",
        r"\bpast\s+(interactions?|communications?)\b",
        r"\bwhat\s+happened\b",
    ],
    "contact_lookup": [
        r"\bwho\s+is\b",
        r"\bcontact\s+(info|details?|information)\b",
        r"\bfind\s+contact\b",
        r"\bget\s+contact\b",
    ],
    "contact_search": [
        r"\b(search|find|list|show)\s+(all\s+)?contacts?\b",
        r"\bcontacts?\s+(at|for|in|from)\b",
        r"\b(decision\s*makers?|champions?|executives?)\b",
        r"\bwho\s+(works|is)\s+(at|with)\b",
        r"\b(key|primary)\s+contacts?\b",
    ],
    "company_search": [
        r"\b(search|find|list|show)\s+(all\s+)?(compan(y|ies)|accounts?)\b",
        r"\bcompan(y|ies)\s+(in|with|from)\b",
        r"\b(enterprise|smb|mid-?market)\s+(accounts?|compan(y|ies))\b",
        r"\bcompan(y|ies)\s+(by|with)\s+(industry|segment|region)\b",
        r"\bwhich\s+compan(y|ies)\b",
        r"\b(in\s+the\s+)?(\w+)\s+industry\b",
    ],
    "groups": [
        r"\bgroup(s)?\b",
        r"\b(at\s*risk|champion|churned)\s+(accounts?|contacts?)\s+group\b",
        r"\bwho\s+is\s+(in|on)\b.*\bgroup\b",
        r"\bmembers?\s+(of|in)\b.*\bgroup\b",
        r"\bgroup\s+(list|members?)\b",
    ],
    "attachments": [
        r"\battachment(s)?\b",
        r"\bdocument(s)?\b",
        r"\bfile(s)?\b",
        r"\bproposal(s)?\b",
        r"\bcontract(s)?\b",
        r"\bpdf(s)?\b",
    ],
}


# =============================================================================
# Router Implementation
# =============================================================================

def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in the text."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def _extract_timeframe(question: str) -> int:
    """Extract days timeframe from question."""
    question_lower = question.lower()
    
    for pattern, extractor in TIMEFRAME_PATTERNS:
        if match := re.search(pattern, question_lower):
            return extractor(match)
    
    # Default: 90 for "what's going on" type questions
    if re.search(r"(going\s+on|happening|status|update)", question_lower):
        return 90
    
    return 30  # Default


def _extract_company_reference(
    question: str, 
    datastore: CRMDataStore | None = None
) -> str | None:
    """
    Try to extract a company name or ID from the question.
    
    Returns the resolved company_id if found.
    """
    ds = datastore or get_datastore()
    
    # Skip extraction if question is clearly about the CRM product itself
    product_mentions = [
        r"\bacme\s+crm\b",          # "Acme CRM" product name
        r"\bthe\s+(crm|system|platform)\b",
        r"\bwhat\s+is\s+(a|an)\b",  # Definition questions
        r"\bhow\s+(do|can|to)\b.*\b(crm|system)\b",
    ]
    question_lower = question.lower()
    for pattern in product_mentions:
        if re.search(pattern, question_lower):
            return None
    
    # Common patterns for company references
    patterns = [
        r"(?:for|with|about|at)\s+([A-Z][A-Za-z0-9\s\-&]+?)(?:\s+(?:in|over|during|last|this|,|\?))",
        r"([A-Z][A-Za-z0-9\s\-&]+?)(?:'s|'s)\s+(?:pipeline|activities|status|renewals?|account)",
        r"(?:company|account|client)\s+(?:called|named)?\s*['\"]?([A-Z][A-Za-z0-9\s\-&]+?)['\"]?(?:\s|\?|$)",
        r"^([A-Z][A-Za-z0-9\s\-&]+?)(?:\s+-|\s+–|\s*:)",  # Company at start
    ]
    
    candidates = []
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            candidate = match.group(1).strip()
            # Clean up trailing words that aren't part of the name
            candidate = re.sub(r"\s+(in|the|this|last|over|during)$", "", candidate, flags=re.I)
            
            if len(candidate) >= 3:  # Minimum name length
                company_id = ds.resolve_company_id(candidate)
                if company_id:
                    return company_id
                candidates.append(candidate)
    
    # Try to find any company name mentioned in the question
    # Get all company names and check if any appear in the question
    ds._build_company_cache()
    question_lower = question.lower()
    
    # Sort by name length (longer names first) to avoid partial matches
    sorted_names = sorted(
        ds._company_names_cache.keys(), 
        key=lambda x: len(x), 
        reverse=True
    )
    
    for name in sorted_names:
        if name in question_lower:
            return ds._company_names_cache[name]
    
    # If no exact match, try partial name matching
    # Look for company name parts mentioned in question
    for name, cid in ds._company_names_cache.items():
        # Check if first word of company name is in question
        first_word = name.split()[0] if name else ""
        if first_word and len(first_word) >= 4 and first_word in question_lower:
            # Verify it's not a common word
            if first_word not in ["green", "delta", "crown", "harbor"]:
                return cid
            # For common words, check for two-word match
            name_parts = name.split()
            if len(name_parts) >= 2:
                two_words = f"{name_parts[0]} {name_parts[1]}"
                if two_words in question_lower:
                    return cid
    
    return None


def _detect_intent(question: str) -> str:
    """Detect the primary intent of the question."""
    question_lower = question.lower()
    scores = {intent: _count_pattern_matches(question_lower, patterns) 
              for intent, patterns in INTENT_PATTERNS.items()}
    
    # Specific intents should win ties over generic ones
    # e.g., "pipeline_summary" over "pipeline", "contact_lookup" over "contact_search"
    INTENT_PRIORITY = {
        "pipeline_summary": 10,
        "deals_at_risk": 10,  # "at risk" should beat generic patterns
        "forecast": 10,  # "forecast" should beat pipeline
        "contact_lookup": 5,
        "groups": 10,  # "who is in the group" should beat "who is"
        "company_status": 5,
        "pipeline": 1,
        "contact_search": 1,
        "company_search": 1,
    }
    
    # Weight scores by priority (add small bonus for specific intents)
    weighted_scores = {
        intent: score + (INTENT_PRIORITY.get(intent, 0) * 0.1 if score > 0 else 0)
        for intent, score in scores.items()
    }
    
    max_intent = max(weighted_scores, key=weighted_scores.get, default="general")
    return max_intent if scores.get(max_intent, 0) > 0 else "general"


def route_question(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    datastore: CRMDataStore | None = None
) -> RouterResult:
    """
    Route a question to determine mode and extract parameters.
    
    Args:
        question: The user's question
        mode: Explicit mode ("auto", "docs", "data", "data+docs")
        company_id: Pre-specified company ID (if any)
        datastore: Optional datastore instance
        
    Returns:
        RouterResult with mode_used, company_id, days, and intent
    """
    question_lower = question.lower()
    
    # If mode is explicitly set (not auto), use it
    if mode and mode != "auto":
        resolved_company = company_id
        if not resolved_company:
            resolved_company = _extract_company_reference(question, datastore)
        
        return RouterResult(
            mode_used=mode,
            company_id=resolved_company,
            days=_extract_timeframe(question),
            intent=_detect_intent(question),
        )
    
    # Auto-detection using heuristics
    docs_score = _count_pattern_matches(question_lower, DOCS_PATTERNS)
    data_score = _count_pattern_matches(question_lower, DATA_PATTERNS)
    
    # Check if there's a company reference
    resolved_company = company_id
    company_name_query = None
    
    if not resolved_company:
        resolved_company = _extract_company_reference(question, datastore)
    
    # If we found a company reference, boost data score
    if resolved_company:
        data_score += 2
    
    # Determine mode
    if docs_score > data_score + 1 and not resolved_company:
        mode_used = "docs"
    elif data_score > docs_score + 1:
        mode_used = "data"
    else:
        # Ambiguous or balanced - use both
        mode_used = "data+docs"
    
    # Extract timeframe
    days = _extract_timeframe(question)
    
    # Detect intent
    intent = _detect_intent(question)
    
    return RouterResult(
        mode_used=mode_used,
        company_id=resolved_company,
        company_name_query=company_name_query,
        days=days,
        intent=intent,
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Router")
    print("=" * 60)
    
    test_questions = [
        # Company-specific
        "What's going on with Acme Manufacturing in the last 90 days?",
        "Show the open pipeline for Beta Tech Solutions",
        "What happened with Crown Foods last month?",
        
        # Documentation
        "How do I create a new opportunity?",
        "What is an Activity in Acme CRM?",
        "How can I import contacts?",
        
        # Renewals
        "Which accounts have upcoming renewals in the next 90 days?",
        
        # Pipeline summary (aggregate)
        "What's the total pipeline value across all accounts?",
        "How many deals are in the pipeline?",
        
        # Contacts
        "Who are the decision makers at our accounts?",
        "Find contacts at Acme Manufacturing",
        "List all champions",
        
        # Companies
        "Show me enterprise accounts",
        "Which companies are in the software industry?",
        
        # Groups
        "Who is in the at-risk accounts group?",
        "List all groups",
        
        # Attachments
        "Find all proposals",
        "Show me contracts for Acme Manufacturing",
        
        # Activities
        "List all deals closing this quarter",
        "What meetings happened last week?",
    ]
    
    for q in test_questions:
        result = route_question(q)
        print(f"\nQ: {q}")
        print(f"   Mode: {result.mode_used}")
        print(f"   Company: {result.company_id}")
        print(f"   Days: {result.days}")
        print(f"   Intent: {result.intent}")
