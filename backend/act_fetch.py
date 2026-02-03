"""Act! API fetch for demo mode."""

from __future__ import annotations

import base64
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
ACT_API_DB = os.getenv("ACT_API_DATABASE", "")

# Timeout for API calls
TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# Demo starters - the 5 fixed questions for demo mode
DEMO_STARTERS = [
    "Brief me on my next call",
    "What should I focus on today?",
    "Who should I contact next?",
    "What's urgent?",
    "Catch me up",
]

# Token cache
_token: str | None = None
_token_expires: float | None = None


def _clear_token() -> None:
    """Clear cached token (call on 401)."""
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
                "Act-Database-Name": ACT_API_DB,
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
        logger.info("Act! API authentication successful")
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
                "Act-Database-Name": ACT_API_DB,
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
        if q == "Brief me on my next call":
            # Fetch next scheduled activity (meeting/call) that hasn't been cleared
            data = _get("/api/activities", {
                "$top": 5,
                "$orderby": "startTime asc",
                "$filter": "isCleared eq false",
            })
            return {"data": data, "error": None}

        elif q == "What should I focus on today?":
            # Fetch today's activities/tasks
            today = time.strftime("%Y-%m-%d")
            data = _get("/api/activities", {
                "$filter": f"startTime ge {today}T00:00:00Z and startTime lt {today}T23:59:59Z",
                "$orderby": "startTime asc",
                "$top": 10,
            })
            return {"data": data, "error": None}

        elif q == "Who should I contact next?":
            # Fetch contacts ordered by last edit (proxy for needing follow-up)
            data = _get("/api/contacts", {
                "$orderby": "edited asc",
                "$top": 5,
            })
            return {"data": data, "error": None}

        elif q == "What's urgent?":
            # Fetch high-priority activities that aren't cleared
            data = _get("/api/activities", {
                "$filter": "activityPriorityName eq 'High' and isCleared eq false",
                "$orderby": "startTime asc",
                "$top": 10,
            })
            return {"data": data, "error": None}

        elif q == "Catch me up":
            # Fetch recent activities ordered by edit time
            data = _get("/api/activities", {
                "$top": 10,
                "$orderby": "edited desc",
            })
            return {"data": data, "error": None}

        return {"data": [], "error": "Unknown question"}

    except Exception as e:
        logger.error(f"act_fetch failed for '{q}': {e}")
        return {"data": [], "error": str(e)}


__all__ = ["DEMO_MODE", "DEMO_STARTERS", "ACT_API_DB", "act_fetch"]
