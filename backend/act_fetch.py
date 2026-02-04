"""Act! API fetch for demo mode."""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Config from environment
DEMO_MODE = os.getenv("ACME_DEMO_MODE", "").lower() in ("true", "1")
ACT_API_URL = os.getenv("ACT_API_BASE_URL", "")
ACT_API_USER = os.getenv("ACT_API_USERNAME", "")
ACT_API_PASS = os.getenv("ACT_API_PASSWORD", "")
_DEFAULT_DATABASE = os.getenv("ACT_API_DATABASE", "")

# Runtime database (mutable) - defaults to env value
_current_database: str = _DEFAULT_DATABASE

# Timeout for API calls (increased for KQC which has 230K contacts)
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# Demo starters - the 5 fixed questions for demo mode
DEMO_STARTERS = [
    "What should I follow up on?",
    "What's coming up?",
    "Who should I contact next?",
    "What needs attention?",
    "What deals are closing soon?",
]

# Custom prompts for each demo question (used by answer/action nodes)
DEMO_PROMPTS = {
    "What should I follow up on?": {
        "answer": "Based on recent CRM history, identify items that need follow-up response. Focus on meetings needing recap, calls with commitments made, and emails requiring replies.",
        "action": "Suggest a specific follow-up action for the most important item. Include who to contact and what to do.",
    },
    "What's coming up?": {
        "answer": "Summarize upcoming scheduled activities from the calendar. Highlight important meetings and their purposes.",
        "action": "Suggest how to prepare for the most important upcoming activity.",
    },
    "Who should I contact next?": {
        "answer": "Prioritize contacts based on opportunity value and how long since last contact. Focus on high-value opportunities that haven't been touched recently.",
        "action": "Recommend a specific outreach action for the top priority contact, including what to discuss.",
    },
    "What needs attention?": {
        "answer": "Flag overdue activities and stale deals that haven't been updated recently. Prioritize by urgency and value.",
        "action": "Recommend which item to address first and what specific action to take.",
    },
    "What deals are closing soon?": {
        "answer": "List deals by expected close date, showing value and probability. Focus on deals closing in the next 30 days.",
        "action": "Suggest a specific next step to help close the most promising deal.",
    },
}

# Available databases for runtime switching
AVAILABLE_DATABASES = ["KQC", "W31322003119"]

# Token cache
_token: str | None = None
_token_expires: float | None = None


def get_database() -> str:
    """Get current database name."""
    return _current_database


def set_database(database: str) -> None:
    """Set current database and clear token cache."""
    global _current_database
    if database not in AVAILABLE_DATABASES:
        raise ValueError(f"Unknown database: {database}. Available: {AVAILABLE_DATABASES}")
    _current_database = database
    _clear_token()  # Clear token when switching databases
    logger.info(f"Switched to database: {database}")


def _clear_token() -> None:
    """Clear cached token (call on 401 or database switch)."""
    global _token, _token_expires
    _token = None
    _token_expires = None


def _get_auth_header() -> str:
    """Get Basic auth header value."""
    credentials = f"{ACT_API_USER}:{ACT_API_PASS}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=2), reraise=True)
def _auth() -> str:
    """Get bearer token with refresh handling."""
    global _token, _token_expires

    # Check if token exists and not expired
    if _token and _token_expires and time.time() < _token_expires:
        return _token

    # Get new token via /authorize endpoint
    try:
        resp = httpx.get(
            f"{ACT_API_URL}/authorize",
            headers={
                "Authorization": _get_auth_header(),
                "Act-Database-Name": _current_database,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()

        # API returns JWT token as raw text, not JSON
        text = resp.text.strip()

        # Check if it's a JWT (starts with eyJ)
        if text.startswith("eyJ"):
            token = text
        else:
            # Try JSON format as fallback
            try:
                data = resp.json()
                token = data.get("token") or data.get("access_token") or data.get("Token")
                if not token:
                    raise ValueError(f"No token in auth response: {data}")
            except Exception:
                raise ValueError(f"Unexpected auth response format: {text[:100]}") from None

        _token = token
        # Set expiry (default 1 hour, adjust based on actual API)
        _token_expires = time.time() + 3600
        logger.info(f"Act! API authentication successful for database: {_current_database}")
        return _token

    except httpx.HTTPError as e:
        logger.error(f"Act! API auth failed: {e}")
        raise


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=2), reraise=True)
def _get(endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """GET from Act! API with error handling and retry."""
    try:
        token = _auth()
        resp = httpx.get(
            f"{ACT_API_URL}{endpoint}",
            headers={
                "Authorization": f"Bearer {token}",
                "Act-Database-Name": _current_database,
            },
            params=params,
            timeout=TIMEOUT,
        )

        # Handle 401 by clearing token (retry will re-auth)
        if resp.status_code == 401:
            _clear_token()
            resp.raise_for_status()

        resp.raise_for_status()
        data = resp.json()

        # API might return data in different structures
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Handle OData response format
            value = data.get("value") or data.get("items") or [data]
            return value if isinstance(value, list) else [value]
        return [data] if data else []

    except httpx.TimeoutException:
        logger.error(f"Act! API timeout calling {endpoint}")
        raise
    except httpx.HTTPError as e:
        logger.error(f"Act! API error: {e}")
        raise


def act_fetch(question: str) -> dict[str, Any]:
    """Fetch data from Act! API for the 5 demo questions.

    Returns: {"data": [...], "error": None} or {"data": [], "error": "message"}
    """
    q = question.strip()

    try:
        if q == "What should I follow up on?":
            # Fetch recent history entries (exclude boring "Field changed" entries)
            data = _get("/api/history", {
                "$top": 50,
                "$orderby": "created desc",
            })
            # Filter out "Field changed" entries which aren't informative
            data = [h for h in data if "Field changed" not in h.get("regarding", "")]
            return {"data": data[:10], "error": None}

        elif q == "What's coming up?":
            # Fetch upcoming activities using calendar endpoint (activities times out on KQC)
            today = time.strftime("%Y-%m-%d", time.gmtime())
            try:
                data = _get("/api/calendar", {
                    "startDate": today,
                    "$top": 10,
                })
            except Exception as e:
                # Fallback to activities with smaller limit if calendar fails
                logger.warning(f"Calendar endpoint failed, trying activities: {e}")
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                data = _get("/api/activities", {
                    "$filter": f"startTime ge {now} and isCleared eq false",
                    "$orderby": "startTime asc",
                    "$top": 5,
                })
            return {"data": data, "error": None}

        elif q == "Who should I contact next?":
            # Fetch high-value opportunities, then get linked contacts
            # Note: Real schema has productTotal/weightedTotal, not estimatedValue
            opps = _get("/api/opportunities", {
                "$orderby": "weightedTotal desc",
                "$top": 15,
            })

            # Extract contact IDs from opportunities
            # Real schema: opportunities have "contacts" array with contact objects
            contact_ids = set()
            for opp in opps:
                # Contacts is an array of objects with "id" field
                contacts_list = opp.get("contacts", [])
                for contact in contacts_list:
                    if isinstance(contact, dict) and contact.get("id"):
                        contact_ids.add(contact["id"])

            # Fetch full contact details if we have IDs
            contacts = []
            if contact_ids:
                for contact_id in list(contact_ids)[:10]:
                    try:
                        contact_data = _get(f"/api/contacts/{contact_id}")
                        if contact_data:
                            contacts.append(contact_data[0] if isinstance(contact_data, list) else contact_data)
                    except Exception:
                        pass

            # Combine opportunities and contacts data
            return {"data": {"opportunities": opps, "contacts": contacts}, "error": None}

        elif q == "What needs attention?":
            # Fetch overdue activities and stale opportunities
            today = time.strftime("%Y-%m-%d", time.gmtime())

            # Try calendar for overdue items, fall back to activities
            overdue = []
            try:
                overdue = _get("/api/calendar", {
                    "endDate": today,
                    "$top": 10,
                })
                # Filter to incomplete items
                overdue = [a for a in overdue if not a.get("isCleared", False)]
            except Exception as e:
                logger.warning(f"Calendar endpoint failed for overdue, trying activities: {e}")
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with contextlib.suppress(Exception):
                    overdue = _get("/api/activities", {
                        "$filter": f"startTime lt {now} and isCleared eq false",
                        "$orderby": "startTime desc",
                        "$top": 5,
                    })

            # Fetch stale opportunities (not touched in 30+ days)
            stale_opps = _get("/api/opportunities", {
                "$orderby": "edited asc",
                "$top": 10,
            })

            return {"data": {"overdue_activities": overdue, "stale_opportunities": stale_opps}, "error": None}

        elif q == "What deals are closing soon?":
            # Fetch opportunities sorted by estimated close date
            # Real schema: estimatedCloseDate (not closeDate)
            data = _get("/api/opportunities", {
                "$orderby": "estimatedCloseDate asc",
                "$top": 15,
            })
            return {"data": data, "error": None}

        return {"data": [], "error": "Unknown question"}

    except Exception as e:
        logger.error(f"act_fetch failed for '{q}': {e}")
        return {"data": [], "error": str(e)}


# Legacy export for backwards compatibility
ACT_API_DB = _DEFAULT_DATABASE

__all__ = [
    "DEMO_MODE",
    "DEMO_STARTERS",
    "DEMO_PROMPTS",
    "ACT_API_DB",
    "AVAILABLE_DATABASES",
    "get_database",
    "set_database",
    "act_fetch",
]
