"""Email generation from CRM interaction history.

Two-step flow:
1. get_contacts_for_category: Fetch history → LLM classifies → Return contacts with reasons
2. generate_email: Fetch contact by ID → Get history from cache → LLM generates email
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from backend.act_fetch import _get
from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)

# LLM model for email generation (gpt-5.2 - gpt-5 has issues with JSON in prompts)
_EMAIL_MODEL = "gpt-5.2"

# Category definitions
CATEGORY_DESCRIPTIONS = {
    "quotes": "Contacts with pending quotes, proposals, or pricing discussions that need follow-up",
    "support": "Contacts with unresolved support issues, problems, or errors that need follow-up",
    "renewals": "Contacts with upcoming renewals, expiring subscriptions, or renewal discussions",
    "recent": "Contacts with recent interactions that may need follow-up",
    "technical": "Contacts with technical issues (sync, database, server, install, upgrade) that need follow-up",
}

# Questions for each category
EMAIL_QUESTIONS = [
    {"id": "quotes", "label": "Who has open quotes that need follow-up?"},
    {"id": "support", "label": "Who needs support follow-up?"},
    {"id": "renewals", "label": "Who should be contacted about renewals?"},
    {"id": "recent", "label": "Who was recently contacted?"},
    {"id": "technical", "label": "Who has technical issues to resolve?"},
]

# Session cache (5 min TTL)
_history_cache: list[dict[str, Any]] = []
_history_cache_time: float = 0
_history_by_contact: dict[str, list[dict[str, Any]]] = {}
_classification_cache: dict[str, list[dict[str, Any]]] = {}
HISTORY_TTL = 300  # 5 minutes


def _is_cache_valid() -> bool:
    """Check if history cache is still valid."""
    return bool(_history_cache) and (time.time() - _history_cache_time) < HISTORY_TTL


def _clear_cache() -> None:
    """Clear all caches."""
    global _history_cache, _history_cache_time, _history_by_contact, _classification_cache
    _history_cache = []
    _history_cache_time = 0
    _history_by_contact = {}
    _classification_cache = {}


def strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"&nbsp;", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


def _relative_time(date_str: str) -> str:
    """Convert date string to relative time like '3 weeks ago'."""
    if not date_str:
        return "unknown"
    try:
        date_part = date_str[:10]
        date_ts = time.mktime(time.strptime(date_part, "%Y-%m-%d"))
        now_ts = time.time()
        days = int((now_ts - date_ts) / 86400)

        if days < 0:
            return "in the future"
        elif days == 0:
            return "today"
        elif days == 1:
            return "yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 14:
            return "1 week ago"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} weeks ago"
        elif days < 60:
            return "1 month ago"
        elif days < 365:
            months = days // 30
            return f"{months} months ago"
        else:
            years = days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
    except Exception:
        return "unknown"


def _condense_history_for_llm(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Condense history records for LLM input (reduce tokens)."""
    condensed = []
    for h in history:
        contacts = h.get("contacts") or []
        if not contacts:
            continue

        # Get first contact info
        contact = contacts[0] if contacts else {}
        contact_id = contact.get("id") if isinstance(contact, dict) else contact
        contact_name = contact.get("displayName", "") if isinstance(contact, dict) else ""

        # Get condensed details (use `or ""` to handle None values)
        details = strip_html(h.get("details") or "")[:500]
        regarding = (h.get("regarding") or "")[:100]
        date = str(h.get("startTime") or "")[:10]

        if contact_id:
            condensed.append({
                "contactId": contact_id,
                "name": contact_name,
                "date": date,
                "regarding": regarding,
                "details": details,
            })

    return condensed


async def _fetch_history() -> list[dict[str, Any]]:
    """Fetch and cache history (1 API call, 5 min TTL)."""
    global _history_cache, _history_cache_time, _history_by_contact

    if _is_cache_valid():
        logger.debug("Using cached history (%d records)", len(_history_cache))
        return _history_cache

    logger.info("Fetching history from Act! API...")
    history = _get("/api/history", {"$top": 1000, "$orderby": "startTime desc"})
    _history_cache = history
    _history_cache_time = time.time()

    # Build per-contact index
    _history_by_contact = {}
    for h in history:
        for c in h.get("contacts") or []:
            cid = c.get("id") if isinstance(c, dict) else c
            if cid:
                _history_by_contact.setdefault(str(cid), []).append(h)

    logger.info("Cached %d history records, %d unique contacts", len(history), len(_history_by_contact))
    return history


class ClassificationResult(BaseModel):
    """LLM classification result."""

    contacts: list[dict[str, Any]]


_CLASSIFICATION_PROMPT = """Analyze these CRM interaction records and identify contacts who need {category} follow-up.

Category: {category_description}

History records:
{history_json}

For each qualifying contact, return:
- contactId: The contact's ID (must match exactly from the input)
- name: Contact's display name
- company: Company name if available (from the interaction context)
- reason: 1 sentence explaining WHY they need follow-up (be specific about the issue/topic)
- lastContact: The date of their last interaction (YYYY-MM-DD format)

Return a JSON object with a "contacts" array. Only include contacts that clearly match the category. Be selective - quality over quantity. Maximum 10 contacts.

Example response:
{{"contacts": [{{"contactId": "abc123", "name": "John Smith", "company": "Acme Corp", "reason": "Requested quote for 50 licenses, awaiting response", "lastContact": "2025-01-15"}}]}}
"""


async def _classify_history_with_llm(history: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    """Send condensed history to LLM for classification."""
    # Check classification cache
    if category in _classification_cache:
        logger.debug("Using cached classification for %s", category)
        return _classification_cache[category]

    condensed = _condense_history_for_llm(history)
    if not condensed:
        return []

    # Limit to avoid token overflow (take most recent 200)
    condensed = condensed[:200]

    category_desc = CATEGORY_DESCRIPTIONS.get(category, category)

    logger.info("Classifying %d history records for category: %s", len(condensed), category)

    chain = create_openai_chain(
        system_prompt="You are a CRM assistant that analyzes interaction history to identify contacts needing follow-up. Always respond with valid JSON.",
        human_prompt=_CLASSIFICATION_PROMPT,
        model=_EMAIL_MODEL,
        max_tokens=2000,
        streaming=False,
    )

    try:
        result = chain.invoke({
            "category": category,
            "category_description": category_desc,
            "history_json": json.dumps(condensed, indent=2),
        })

        # Parse JSON response
        logger.debug("LLM result type: %s, value: %s", type(result), result[:200] if result else "None")
        if not result:
            logger.warning("LLM returned empty response for category: %s", category)
            return []

        # Handle potential markdown code blocks
        result_text = result.strip()
        if result_text.startswith("```"):
            # Remove markdown code block
            result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
            result_text = re.sub(r"\s*```$", "", result_text)

        parsed = json.loads(result_text)
        contacts = parsed.get("contacts", [])

        # Add lastContactAgo field
        for c in contacts:
            c["lastContactAgo"] = _relative_time(c.get("lastContact", ""))

        # Cache result
        _classification_cache[category] = contacts
        logger.info("Classified %d contacts for category: %s", len(contacts), category)
        return contacts

    except Exception as e:
        logger.error("LLM classification failed: %s", e)
        return []


async def get_contacts_for_category(category: str) -> list[dict[str, Any]]:
    """Get contacts for a category.

    1. Fetch history (1 API call, cached)
    2. LLM classifies all history records for this category
    3. Returns contacts with AI-generated reasons
    """
    if category not in CATEGORY_DESCRIPTIONS:
        raise ValueError(f"Unknown category: {category}")

    # Fetch history (cached)
    history = await _fetch_history()

    # Filter history with details only
    history_with_details = [h for h in history if h.get("details")]

    # LLM classification
    contacts = await _classify_history_with_llm(history_with_details, category)

    # Sort by lastContact ascending (oldest first = most likely needs follow-up)
    contacts.sort(key=lambda x: x.get("lastContact", "9999-99-99"))

    return contacts[:10]  # Max 10 contacts


_EMAIL_PROMPT = """Generate a professional follow-up email for {contact_name} at {company}.

Context: This is a {category} follow-up.

Recent interaction ({date}):
---
Subject: {regarding}
Details: {details}
---

Requirements:
1. Reference the specific interaction context
2. Keep it concise (3-5 sentences)
3. Include a clear call to action
4. Professional but friendly tone
5. No signature (user will add their own)
6. Do not include "Subject:" in the body

Return a JSON object with "subject" and "body" fields.
Example: {{"subject": "Following up on your quote request", "body": "Hi John,\\n\\nI wanted to follow up on..."}}
"""


def build_mailto_link(email: str, subject: str, body: str) -> str:
    """Create mailto: URI with URL-encoded params."""
    encoded_subject = quote(subject, safe="")
    encoded_body = quote(body, safe="")
    return f"mailto:{email}?subject={encoded_subject}&body={encoded_body}"


async def generate_email(contact_id: str, category: str) -> dict[str, Any]:
    """Generate email for a specific contact.

    1. Fetch contact by ID (1 API call) - to get email
    2. Get history from cache
    3. LLM generates email
    Returns {subject, body, mailtoLink, contact}
    """
    # Fetch contact by ID
    logger.info("Fetching contact %s for email generation", contact_id)
    try:
        result = _get(f"/api/contacts/{contact_id}", {})
        contact = result[0] if isinstance(result, list) and result else result
    except Exception as e:
        logger.error("Failed to fetch contact %s: %s", contact_id, e)
        raise ValueError(f"Could not fetch contact: {e}") from e

    email_address = contact.get("emailAddress")
    if not email_address:
        raise ValueError(f"Contact {contact_id} has no email address")

    contact_name = contact.get("fullName", "")
    company = contact.get("company", "")

    # Get history from cache
    history_list = _history_by_contact.get(str(contact_id), [])
    if not history_list:
        # Fetch history if not in cache
        await _fetch_history()
        history_list = _history_by_contact.get(str(contact_id), [])

    # Get most recent history item with details
    recent_history = None
    for h in history_list:
        if h.get("details"):
            recent_history = h
            break

    if not recent_history:
        # Generate generic email without specific context
        regarding = "your account"
        details = "I wanted to reach out regarding our ongoing relationship."
        date = "recently"
    else:
        regarding = recent_history.get("regarding", "your recent inquiry")
        details = strip_html(recent_history.get("details", ""))[:500]
        date = str(recent_history.get("startTime", ""))[:10]

    # Generate email with LLM
    chain = create_openai_chain(
        system_prompt="You are a professional email writer for sales follow-ups. Always respond with valid JSON.",
        human_prompt=_EMAIL_PROMPT,
        model=_EMAIL_MODEL,
        max_tokens=500,
        streaming=False,
    )

    try:
        result_text = chain.invoke({
            "contact_name": contact_name or "there",
            "company": company or "your company",
            "category": CATEGORY_DESCRIPTIONS.get(category, category),
            "date": date,
            "regarding": regarding,
            "details": details,
        })

        # Parse JSON response
        result_text = result_text.strip()
        if result_text.startswith("```"):
            result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
            result_text = re.sub(r"\s*```$", "", result_text)

        parsed = json.loads(result_text)
        subject = parsed.get("subject", "Follow-up")
        body = parsed.get("body", "")

        return {
            "subject": subject,
            "body": body,
            "mailtoLink": build_mailto_link(email_address, subject, body),
            "contact": {
                "id": contact_id,
                "name": contact_name,
                "email": email_address,
                "company": company,
            },
        }

    except Exception as e:
        logger.error("LLM email generation failed: %s", e)
        raise ValueError(f"Failed to generate email: {e}") from e


def get_questions() -> list[dict[str, str]]:
    """Return the 5 question categories."""
    return EMAIL_QUESTIONS


__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "EMAIL_QUESTIONS",
    "get_questions",
    "get_contacts_for_category",
    "generate_email",
    "strip_html",
    "build_mailto_link",
]
