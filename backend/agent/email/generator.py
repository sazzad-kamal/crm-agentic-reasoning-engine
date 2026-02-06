"""Email generation from CRM interaction history.

Two-step flow:
1. get_contacts_for_category: Fetch history → LLM classifies → Return contacts with reasons
2. generate_email: Fetch contact by ID → Get history from cache → LLM generates email
"""

from __future__ import annotations

import asyncio
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
    "support": "Contacts with unresolved support issues, problems, or errors that need follow-up",
    "renewals": "Contacts with upcoming renewals, expiring subscriptions, or renewal discussions",
    "billing": "Contacts with invoice, payment, or billing discussions that need follow-up",
    "quotes": "Contacts with pending quotes, proposals, or pricing discussions that need follow-up",
}

# Tone guidance per category (used in email generation)
CATEGORY_TONES: dict[str, tuple[str, str]] = {
    # (tone_name, tone_instruction)
    "quotes": ("professional", "Be professional and business-focused. Emphasize value and next steps."),
    "billing": ("professional", "Be professional and tactful. Acknowledge the sensitive nature of payment discussions."),
    "support": ("empathetic", "Be warm and understanding. Show you care about resolving their issue."),
    "renewals": ("friendly", "Be friendly and appreciative. Emphasize the value of the ongoing relationship."),
}

# Questions for each category
EMAIL_QUESTIONS: list[dict[str, str]] = [
    {"id": "support", "label": "Who needs support follow-up?"},
    {"id": "renewals", "label": "Who should be contacted about renewals?"},
    {"id": "billing", "label": "Who has billing issues to resolve?"},
    {"id": "quotes", "label": "Who has open quotes that need follow-up?"},
]

# Session cache (5 min TTL)
_history_cache: list[dict[str, Any]] = []
_history_cache_time: float = 0
_history_by_contact: dict[str, list[dict[str, Any]]] = {}
_classification_cache: dict[str, list[dict[str, Any]]] = {}
_history_fetch_task: asyncio.Task[list[dict[str, Any]]] | None = None  # Shared task for deduplication
HISTORY_TTL = 300  # 5 minutes


def _is_cache_valid() -> bool:
    """Check if history cache is still valid."""
    return bool(_history_cache) and (time.time() - _history_cache_time) < HISTORY_TTL


def _clear_cache() -> None:
    """Clear all caches."""
    global _history_cache, _history_cache_time, _history_by_contact, _classification_cache, _history_fetch_task
    _history_cache = []
    _history_cache_time = 0
    _history_by_contact = {}
    _classification_cache = {}
    _history_fetch_task = None


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
            # Show actual time for today (fast-moving system)
            if len(date_str) >= 16:  # Has time component like "2026-02-06T14:30"
                try:
                    time_part = date_str[11:16]  # "14:30"
                    hour, minute = int(time_part[:2]), int(time_part[3:5])
                    period = "AM" if hour < 12 else "PM"
                    display_hour = hour if hour <= 12 else hour - 12
                    if display_hour == 0:
                        display_hour = 12
                    return f"today at {display_hour}:{minute:02d} {period}"
                except (ValueError, IndexError):
                    return "today"
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


# History types to exclude from LLM classification
# - Closed opportunities: no follow-up needed
# - Noise types: automated/system records that don't indicate communication
_EXCLUDED_HISTORY_TYPES = {
    # Closed opportunities
    "Opportunity Lost",
    "Opportunity Inactive",
    # Noise types (system/automated, not real communication)
    "Field Changed",
    "Web Activity",
    "Contact Updated",
    "Contact Deleted",
    "Contact Linked",
}


def _is_future_date(date_str: str | None) -> bool:
    """Check if date is in the future (data error)."""
    if not date_str:
        return False
    try:
        date_part = date_str[:10]
        date_ts = time.mktime(time.strptime(date_part, "%Y-%m-%d"))
        return date_ts > time.time()
    except Exception:
        return False


def _get_history_type(h: dict[str, Any]) -> str:
    """Extract history type as a string, handling various API response formats."""
    history_type = h.get("historyType") or h.get("type")
    if history_type is None:
        return ""
    # Handle case where historyType is a dict with a "name" key
    if isinstance(history_type, dict):
        return str(history_type.get("name", ""))
    return str(history_type)


def _filter_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pre-filter history to remove garbage data before LLM sees it.

    Removes:
    - Records with future dates (data errors)
    - Excluded types: closed opportunities and noise (Field Changed, Web Activity, etc.)
    """
    filtered = []
    for h in history:
        # Skip future-dated records
        if _is_future_date(h.get("startTime")):
            continue

        # Skip Opportunity Lost/Inactive
        history_type = _get_history_type(h)
        if history_type in _EXCLUDED_HISTORY_TYPES:
            continue

        filtered.append(h)

    return filtered


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


async def _do_fetch_history() -> list[dict[str, Any]]:
    """Actual fetch implementation (called by shared task)."""
    global _history_cache, _history_cache_time, _history_by_contact

    logger.info("Fetching history from Act! API...")
    history: list[dict[str, Any]] = _get("/api/history", {"$top": 1000, "$orderby": "startTime desc"})
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


async def _fetch_history() -> list[dict[str, Any]]:
    """Fetch and cache history with shared task pattern.

    If a fetch is already in progress, wait for it instead of starting another.
    This prevents duplicate API calls when warmup and user click overlap.

    Note: Act! API doesn't support $filter on /api/history endpoint,
    so filtering is done in Python via _filter_history().
    """
    global _history_fetch_task

    # Return cached if valid
    if _is_cache_valid():
        logger.debug("Using cached history (%d records)", len(_history_cache))
        return _history_cache

    # If fetch already in progress, wait for it (don't start another)
    if _history_fetch_task is not None and not _history_fetch_task.done():
        logger.debug("Waiting for in-flight history fetch task...")
        return await _history_fetch_task

    # Start new fetch task
    _history_fetch_task = asyncio.create_task(_do_fetch_history())
    return await _history_fetch_task


def get_cache_age() -> int | None:
    """Return seconds since cache was populated, or None if empty."""
    if not _history_cache or not _history_cache_time:
        return None
    return int(time.time() - _history_cache_time)


async def warmup_cache() -> int:
    """Prefetch history to warm cache. Returns record count."""
    history = await _fetch_history()
    return len(history)


class ClassificationResult(BaseModel):
    """LLM classification result."""

    contacts: list[dict[str, Any]]


_CLASSIFICATION_PROMPT = """Analyze CRM interaction records and identify contacts who need {category} follow-up.

Category: {category_description}

Contact history:
{history_json}

## Interaction Type Guide

**OUTBOUND (we contacted them - ball is with them):**
- E-mail Sent, Knowtifier Email Sent
- Call Attempted, Call Left Message, Call Completed, Sales Call - Completed

**INBOUND (they contacted us - ball is with us):**
- Call Received, E-mail Auto Attached

**MEETINGS (two-way):**
- Meeting Held, Appointment Completed

**ADMINISTRATIVE (ignore for follow-up decisions):**
- To-do Done, Attachment, Field Changed, Contact Updated, Contact Deleted, Contact Linked, Web Activity

Determine WHO HAS THE BALL: If last meaningful interaction was OUTBOUND = waiting on them. If INBOUND = waiting on us.

## EXCLUSION RULES - Skip contacts who:
1. Have status: Closed-Lost, Declined, Not Ready, Quote Disabled, Cancelled, Terminated
2. Explicitly said "not moving forward", "not interested", "decided against", or "going with competitor"
3. Have their issue already resolved, completed, or marked closed
4. Are post-sale implementation items (not pre-sale opportunities)

## Category-Specific Rules

**quotes**:
- ONLY include contacts with ACTIVE/OPEN/PENDING quote status awaiting a decision
- Include quote ID when available (e.g., "Quote #18177KQC")
- EXCLUDE: Closed-Lost, Not Ready, Quote Disabled, "not moving forward", post-sale/implementation
- Priority: Quotes with stated decision deadlines > quotes awaiting approval > general pending quotes

**billing**:
- Focus: Unpaid invoices, payment issues, collections, chargebacks, suspensions
- Include invoice # when available in the reason
- SORT priority: Overdue/past due > suspended accounts > due this week > due soon > upcoming
- EXCLUDE: Routine renewal reminders with no payment issue (those belong in renewals)

**support**:
- ONLY include unresolved issues with: active errors, problems, or explicit "in progress" status
- Include case/ticket ID when available
- Specify in reason: What's broken + what's needed to resolve
- EXCLUDE: Resolved issues, "monitoring" without active issue, feature requests (not break/fix)

**renewals**:
- Focus: Upcoming expirations without confirmed renewal or payment
- Include: renewal/expiry date, invoice # when available
- SORT by: Expiration date (soonest first)
- EXCLUDE: Already renewed/paid, explicitly declined renewal

## Output Requirements

For each qualifying contact return:
- contactId: Exact ID from input (must match exactly)
- name: Display name
- company: Company name
- reason: 1 sentence with: [Specific situation] + [What's needed next]
  Example: "Quote #18177KQC sent Feb 5 awaiting boss approval - follow up for decision"
  Example: "Invoice #79371 due Mar 3, promised to pay 'tomorrow' - confirm payment received"
- lastContact: YYYY-MM-DD format

Return JSON: {{"contacts": [...]}}

Maximum 10 contacts, sorted by priority/urgency. Quality over quantity - only include contacts that CLEARLY match the category criteria. When in doubt, leave them out.
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
        contacts: list[dict[str, Any]] = parsed.get("contacts", [])

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
    2. Pre-filter to remove garbage data (future dates, closed opportunities)
    3. LLM classifies remaining history records for this category
    4. Returns contacts with AI-generated reasons
    """
    if category not in CATEGORY_DESCRIPTIONS:
        raise ValueError(f"Unknown category: {category}")

    # Fetch history (cached)
    history = await _fetch_history()

    # Filter history with details only
    history_with_details = [h for h in history if h.get("details")]

    # Pre-filter to remove garbage data before LLM sees it
    filtered_history = _filter_history(history_with_details)
    logger.info(
        "Pre-filtered history: %d -> %d records (removed %d)",
        len(history_with_details),
        len(filtered_history),
        len(history_with_details) - len(filtered_history),
    )

    # LLM classification
    contacts = await _classify_history_with_llm(filtered_history, category)

    # Sort by lastContact ascending (oldest first = most likely needs follow-up)
    contacts.sort(key=lambda x: x.get("lastContact", "9999-99-99"))

    return contacts[:10]  # Max 10 contacts


_EMAIL_PROMPT = """Generate a follow-up email for {contact_name} at {company}.

Context: This is a {category} follow-up.

Recent interaction ({date}):
---
Subject: {regarding}
Details: {details}
---

Tone: {tone}
{tone_instruction}

Requirements:
1. Reference the specific interaction context
2. Keep it concise (3-5 sentences)
3. Include a clear call to action
4. No signature (user will add their own)
5. Do not include "Subject:" in the body

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
        contact: dict[str, Any] = result[0] if isinstance(result, list) and result else result  # type: ignore[assignment]
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

    # Get tone for this category
    tone_name, tone_instruction = CATEGORY_TONES.get(category, ("professional", "Be professional and helpful."))

    try:
        result_text = chain.invoke({
            "contact_name": contact_name or "there",
            "company": company or "your company",
            "category": CATEGORY_DESCRIPTIONS.get(category, category),
            "date": date,
            "regarding": regarding,
            "details": details,
            "tone": tone_name,
            "tone_instruction": tone_instruction,
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
    "warmup_cache",
    "get_cache_age",
    "_clear_cache",
]
