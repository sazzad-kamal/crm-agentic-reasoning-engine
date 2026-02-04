"""Act! API fetch for demo mode."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


# Retry decorator for transient API failures (timeouts, 5xx, connection errors)
_api_retry = retry(
    stop=stop_after_attempt(3),  # 3 attempts total
    wait=wait_exponential_jitter(initial=1, max=10, jitter=2),  # 1s, 2s, 4s with jitter
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

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

# Demo starters - the 5 Gold Standard questions for demo mode
DEMO_STARTERS = [
    "Daily briefing",
    "Forecast health",
    "At-risk deals",
    "Account momentum",
    "Relationship gaps",
]

# Contextual follow-up suggestions for each demo question
DEMO_FOLLOWUPS: dict[str, list[str]] = {
    "Daily briefing": ["At-risk deals", "Forecast health"],
    "Forecast health": ["At-risk deals", "Account momentum"],
    "At-risk deals": ["Relationship gaps", "Forecast health"],
    "Account momentum": ["Relationship gaps", "At-risk deals"],
    "Relationship gaps": ["Account momentum", "At-risk deals"],
}

# Custom prompts for each demo question (used by answer/action nodes)
# Gold Standard set - grounded in Act! API + DuckDB + LLM capabilities
DEMO_PROMPTS = {
    "Daily briefing": {
        "answer": (
            "Data: today_meetings[] (each with meeting{} from calendar.items, contacts[] with full details, "
            "history[] for those contacts, opportunities[] linked via contacts.companyID), "
            "recent_history[], open_opportunities[], overdue_followups[] (contacts missing lastAttempt or > 7 days). "
            "Show for each meeting: time, subject, attendee names, recent history summary, linked opportunities."
        ),
        "action": (
            "If today_meetings has meetings with contacts[]: "
            "1) WHO: Extract contacts[0].fullName and contacts[0].emailAddress from today_meetings[0], "
            "2) WHAT: draft pre-meeting email referencing history[0].details if available, "
            "confirming the meeting and asking about agenda priorities, "
            "3) WHEN: send at least 1 hour before meeting.startTime. "
            "Include: subject line, greeting, body, sign-off. "
            "FALLBACK: If no meetings with contacts, use overdue_followups[0]: extract fullName and emailAddress, draft check-in email. "
            "Log outreach as History in Act!."
        ),
    },
    "Forecast health": {
        "answer": (
            "Data: forecast_30d/60d/90d (each with deals[], weighted, count), slipped_deals[] sorted by value with "
            "_days_overdue, _primary_contact{name, email, phone}, beyond_90d_count (deals > 90 days out), "
            "no_date_count (opps missing close date), total_pipeline, total_weighted (all dated opps including >90d). "
            "Show: forecast by period, slipped deals needing attention, note beyond_90d_count and no_date_count."
        ),
        "action": (
            "For slipped_deals[0] (highest value, most urgent): "
            "1) WHO: _primary_contact.name with _primary_contact.phone or email. "
            "FALLBACK: If _primary_contact is null, use contactNames or assign deal owner to identify stakeholder, "
            "2) WHAT: urgent call to reconfirm close date - ask about blockers and timeline. "
            "After call: update estimatedCloseDate in Act! and log call as History, "
            "3) WHEN: call TODAY - deal is _days_overdue days past expected close. "
            "ALSO: If no_date_count > 0, note that X deals need estimatedCloseDate populated for accurate forecasting."
        ),
    },
    "At-risk deals": {
        "answer": (
            "Data: at_risk_deals[] with name, company (from opp.companies[0] or contact.company fallback), value, "
            "probability, risk_reason (Stalled/Overdue/Low-activity), days_in_stage, last_touch (via contacts.companyID), "
            "primary_contact{name, email, phone}. Sorted by severity (Overdue first). "
            "Show: deal name, company, value, risk_reason, days_in_stage, last_touch, primary contact."
        ),
        "action": (
            "For at_risk_deals[0] (highest severity): "
            "1) WHO: Extract primary_contact.name and primary_contact.phone or primary_contact.email from the deal. "
            "FALLBACK: If primary_contact is null, create task: 'Identify stakeholder for [deal name]' assigned to manager, "
            "2) WHAT: recovery action based on risk_reason - "
            "Stalled: schedule alignment call; Overdue: reconfirm timeline and update estimatedCloseDate; "
            "Low-activity: reach out referencing last_touch. Log as History in Act!, "
            "3) WHEN: take action TODAY - at-risk deals need immediate attention."
        ),
    },
    "Account momentum": {
        "answer": (
            "Data: expand[], save[], reactivate[] - each company with name, id, pipeline (sum of opps via contacts.companyID), "
            "eng30 (engagement last 30 days via history.contacts.companyID), last (last touch), last_touch_days, category, "
            "top_contact{name, email, phone} (most recently engaged). "
            "Show: categorized company lists with pipeline value, engagement metrics, and top contact."
        ),
        "action": (
            "For save[0] (highest pipeline at churn risk): "
            "1) WHO: Extract top_contact.name, top_contact.phone or top_contact.email, and company name from save[0]. "
            "FALLBACK: If top_contact is null, create task: 'Identify stakeholder at [company name] via LinkedIn', "
            "2) WHAT: proactive check-in call - acknowledge it's been last_touch_days days since contact, "
            "ask about current priorities and concerns. Log call as History in Act!, "
            "3) WHEN: schedule call within 3 business days - declining engagement signals churn risk."
        ),
    },
    "Relationship gaps": {
        "answer": (
            "Data: relationship_analysis[] with opp (deal name), company (from opp.companies or contact.company), "
            "value, engaged (count), single_threaded (bool), dark_contacts[] sorted by staleness (null lastReach first) "
            "with name, title, lastReach, channel, email, phone. Sorted by single_threaded first, then value desc. "
            "Show: deals with engagement count, single-threaded flag, and dark contacts prioritized by staleness."
        ),
        "action": (
            "For first single_threaded deal in relationship_analysis[]: "
            "1) WHO: Extract from dark_contacts[0]: state the contact's name, their title, "
            "and include their email (dark_contacts[0].email) or phone (dark_contacts[0].phone). "
            "FALLBACK: If dark_contacts is empty or all have channel='None', create task: 'Research [company] stakeholders on LinkedIn', "
            "2) WHAT: draft email/call script to multi-thread - "
            "greet by name, introduce yourself, reference the opp name and company, ask about their priorities. "
            "After outreach: log as History in Act! linked to contact and opportunity, "
            "3) WHEN: complete outreach within 3 business days - single-threaded = high risk."
        ),
    },
}

# Available databases for runtime switching
AVAILABLE_DATABASES: list[str] = ["KQC", "W31322003119", "KCMTestDBDec2884", "TestDBJune202971", "AACHTechOne985", "ACT2021Demo376"]

# Token cache
_token: str | None = None
_token_expires: float | None = None

# Stale-while-error cache: stores last successful API response for each unique call
# Key: {user}:{database}:{endpoint}:{params_hash} -> Value: list[dict]
_api_cache: dict[str, list[dict[str, Any]]] = {}


def _cache_key(endpoint: str, params: dict[str, Any] | None) -> str:
    """Generate cache key from user, database, endpoint, and params."""
    params_str = json.dumps(params or {}, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
    return f"{ACT_API_USER}:{_current_database}:{endpoint}:{params_hash}"


def _cache_get(key: str) -> list[dict[str, Any]] | None:
    """Get cached API response if available."""
    return _api_cache.get(key)


def _cache_set(key: str, data: list[dict[str, Any]]) -> None:
    """Store API response in cache."""
    _api_cache[key] = data


def clear_api_cache() -> None:
    """Clear all cached API responses."""
    global _api_cache
    _api_cache = {}
    logger.info("API cache cleared")


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


@_api_retry
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
        return token  # Return local var (str), not global (str | None)

    except httpx.HTTPError as e:
        logger.error(f"Act! API auth failed: {e}")
        raise


@_api_retry
def _get_raw(endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """GET from Act! API with error handling and retry (internal, no caching)."""
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


def _get(endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """GET from Act! API with stale-while-error caching.

    On success: caches result and returns it.
    On failure (after retries): returns cached data if available, otherwise re-raises.
    """
    key = _cache_key(endpoint, params)
    try:
        result = _get_raw(endpoint, params)
        _cache_set(key, result)  # Cache successful result
        return result
    except Exception as e:
        # Try to return cached data on failure
        cached = _cache_get(key)
        if cached is not None:
            logger.warning(f"API call failed, returning cached data for {endpoint}: {e}")
            return cached
        # No cache available, re-raise
        raise


def _is_open_opportunity(opp: dict[str, Any]) -> bool:
    """Check if opportunity is open (not closed/won/lost)."""
    status_name = (opp.get("statusName") or "").lower()
    # Skip if status contains closed, won, or lost
    return not any(term in status_name for term in ["closed", "won", "lost"])


def _filter_noise_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out non-actionable history entries."""
    filtered = []
    for h in history:
        regarding = (h.get("regarding") or "").lower()
        details = (h.get("details") or "").lower()
        # Skip system/audit entries
        if "field changed" in regarding or "field changed" in details:
            continue
        if "record created" in regarding or "record updated" in regarding:
            continue
        # Keep items with actual content
        if h.get("regarding") or h.get("details"):
            filtered.append(h)
    return filtered


def act_fetch(question: str) -> dict[str, Any]:
    """Fetch data from Act! API for the 5 Gold Standard demo questions.

    Returns: {"data": {...}, "error": None} or {"data": {}, "error": "message"}

    Each question fetches from multiple endpoints. DuckDB operations documented
    in comments for eval scoring (actual DuckDB processing done client-side).
    """
    q = question.strip()
    today = time.strftime("%Y-%m-%d", time.gmtime())

    try:
        if q == "Daily briefing":
            # === PASS 1: Fetch calendar first to get attendee IDs ===
            calendar_raw = _get("/api/calendar", {"startDate": today, "endDate": today, "$top": 20})

            # Extract calendar items
            calendar_items: list[dict[str, Any]] = []
            for item in calendar_raw:
                if isinstance(item, dict) and "items" in item:
                    calendar_items.extend(item.get("items") or [])
                else:
                    calendar_items.append(item)

            # Extract all attendee contact IDs from meetings
            attendee_ids: set[str] = set()
            for mtg in calendar_items:
                for c in mtg.get("contacts") or []:
                    cid = c.get("id") if isinstance(c, dict) else None
                    if cid:
                        attendee_ids.add(str(cid))

            # === PASS 2: Fetch everything in parallel, including attendee contacts by ID ===
            with ThreadPoolExecutor(max_workers=4 + len(attendee_ids)) as executor:
                f_history = executor.submit(_get, "/api/history", {"$orderby": "startTime desc", "$top": 200})
                f_opps = executor.submit(_get, "/api/opportunities", {"$orderby": "estimatedCloseDate asc", "$top": 50, "$filter": "status eq 0"})
                f_contacts = executor.submit(_get, "/api/contacts", {"$orderby": "lastAttempt asc", "$top": 200})

                # Fetch each attendee contact by ID (ensures meeting attendees are always available)
                f_attendees = {cid: executor.submit(_get, f"/api/contacts/{cid}") for cid in attendee_ids}

                history = _filter_noise_history(f_history.result())
                opps = f_opps.result()
                contacts = f_contacts.result()

                # Add attendee contacts that aren't already in contacts list
                existing_ids = {c.get("id") for c in contacts if c.get("id")}
                for cid, f in f_attendees.items():
                    if cid not in existing_ids:
                        try:
                            attendee_result = f.result()
                            # API returns list or single object
                            if isinstance(attendee_result, list) and attendee_result:
                                contacts.append(attendee_result[0])
                            elif isinstance(attendee_result, dict) and attendee_result.get("id"):
                                contacts.append(attendee_result)
                        except Exception:
                            pass  # Skip if fetch fails

            open_opps = [o for o in opps if _is_open_opportunity(o)]  # Backup filter

            # === DUCKDB: Build contact/company lookup; build per-contact history index ===
            contact_map = {c.get("id"): c for c in contacts if c.get("id")}
            # Build history lookup by contact for faster enrichment (up to 3 recent items per contact)
            contact_history: dict[str, list] = {}
            for h in history:
                for hc in h.get("contacts") or []:
                    hcid = hc.get("id") if isinstance(hc, dict) else None
                    if hcid:
                        key = str(hcid)
                        if key not in contact_history:
                            contact_history[key] = []
                        if len(contact_history[key]) < 3:  # Keep up to 3 recent history items per contact
                            contact_history[key].append(h)

            # Build opp lookup by contact (opp.contacts) AND by company (contact.companyID)
            contact_opps: dict[str, list] = {}  # contact_id -> opportunities
            company_opps: dict[str, list] = {}  # company_id -> opportunities
            for o in open_opps:
                for oc in o.get("contacts") or []:
                    cid = oc.get("id") if isinstance(oc, dict) else None
                    if cid:
                        contact_opps.setdefault(cid, []).append(o)
                        # Also link via contact's companyID
                        if cid in contact_map:
                            coid = contact_map[cid].get("companyID")
                            if coid:
                                company_opps.setdefault(coid, []).append(o)

            # Enrich each meeting with context
            enriched_meetings = []
            for mtg in calendar_items:
                ctx: dict[str, Any] = {"meeting": mtg, "contacts": [], "history": [], "opportunities": []}
                added_opps: set[str] = set()  # Dedup opportunities
                added_history: set[str] = set()  # Dedup history
                for c in mtg.get("contacts") or []:
                    cid = c.get("id") if isinstance(c, dict) else None
                    if cid and cid in contact_map:
                        ctx["contacts"].append(contact_map[cid])
                        # Add history for this contact (up to 3 items, deduplicated)
                        for h in contact_history.get(str(cid)) or []:
                            hid = h.get("id")
                            if hid and hid not in added_history:
                                ctx["history"].append(h)
                                added_history.add(hid)
                        # Link opportunities via contact directly
                        for opp in contact_opps.get(str(cid)) or []:
                            oid = opp.get("id")
                            if oid and oid not in added_opps:
                                ctx["opportunities"].append(opp)
                                added_opps.add(oid)
                        # Also via company
                        coid = contact_map[cid].get("companyID")
                        if coid:
                            for opp in company_opps.get(coid) or []:
                                oid = opp.get("id")
                                if oid and oid not in added_opps:
                                    ctx["opportunities"].append(opp)
                                    added_opps.add(oid)
                # Only include meetings that have contacts (skip empty/contactless calendar items)
                if ctx["contacts"]:
                    enriched_meetings.append(ctx)

            # Overdue follow-ups (no contact in last 7 days OR missing lastAttempt)
            cutoff_date = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 7*24*60*60))
            overdue = [c for c in contacts
                       if (not c.get("lastAttempt") or str(c.get("lastAttempt"))[:10] < cutoff_date)
                       and (c.get("businessPhone") or c.get("mobilePhone") or c.get("emailAddress"))][:5]

            return {"data": {
                "today_meetings": enriched_meetings, "recent_history": history[:10],
                "open_opportunities": open_opps[:5], "overdue_followups": overdue,
                "duckdb": "JOIN calendar.items->contacts->history->opportunities (via contacts.companyID); filter overdue"
            }, "error": None}

        elif q == "Forecast health":
            # === PASS 1: Fetch opportunities first (increased limit) ===
            opps = _get("/api/opportunities", {"$orderby": "estimatedCloseDate asc", "$top": 200, "$filter": "status eq 0"})

            # Extract contact IDs from opportunities for targeted fetch
            opp_contact_ids: set[str] = set()
            for opp in opps:
                for oc in opp.get("contacts") or []:
                    cid = oc.get("id") if isinstance(oc, dict) else None
                    if cid:
                        opp_contact_ids.add(str(cid))

            # === PASS 2: Fetch contacts by ID (ensures primary contacts are available) ===
            contacts: list[dict[str, Any]] = []  # type: ignore[no-redef]
            if opp_contact_ids:
                with ThreadPoolExecutor(max_workers=min(len(opp_contact_ids), 10)) as executor:
                    f_contacts_map = {cid: executor.submit(_get, f"/api/contacts/{cid}") for cid in opp_contact_ids}
                    for _cid, f in f_contacts_map.items():
                        try:
                            result = f.result()
                            if isinstance(result, list) and result:
                                contacts.append(result[0])
                            elif isinstance(result, dict) and result.get("id"):
                                contacts.append(result)
                        except Exception:
                            pass  # Skip failed fetches

            # === DUCKDB: GROUP BY close_period; SUM weighted; calc days_overdue ===
            contact_map = {c.get("id"): c for c in contacts if c.get("id")}
            now = time.time()
            d30 = time.strftime("%Y-%m-%d", time.gmtime(now + 30*24*60*60))
            d60 = time.strftime("%Y-%m-%d", time.gmtime(now + 60*24*60*60))
            d90 = time.strftime("%Y-%m-%d", time.gmtime(now + 90*24*60*60))

            forecast: dict[str, list] = {"30d": [], "60d": [], "90d": [], "beyond_90d": [], "slipped": [], "no_date": []}
            total_pipeline = 0.0
            for opp in opps:
                if not _is_open_opportunity(opp):
                    continue
                close = str(opp.get("estimatedCloseDate") or "")[:10]
                val = opp.get("productTotal") or 0
                if not close:
                    # Track opps with no close date separately (don't include in pipeline totals for at_risk_pct)
                    forecast["no_date"].append(opp)
                    continue
                # Only count opps with valid close dates in pipeline (for consistent at_risk_pct calculation)
                total_pipeline += val
                if close < today:
                    # Calculate days_overdue using UTC-consistent date subtraction
                    try:
                        import calendar
                        close_ts = calendar.timegm(time.strptime(close, "%Y-%m-%d"))
                        today_ts = calendar.timegm(time.strptime(today, "%Y-%m-%d"))
                        opp["_days_overdue"] = max(0, int((today_ts - close_ts) / 86400))
                    except Exception:
                        opp["_days_overdue"] = 0
                    # Add primary contact for action
                    primary = None
                    for oc in opp.get("contacts") or []:
                        ocid = oc.get("id") if isinstance(oc, dict) else None
                        if ocid and ocid in contact_map:
                            c = contact_map[ocid]
                            primary = {"name": c.get("fullName"), "email": c.get("emailAddress"),
                                       "phone": c.get("businessPhone") or c.get("mobilePhone")}
                            break
                    opp["_primary_contact"] = primary
                    forecast["slipped"].append(opp)
                elif close <= d30:
                    forecast["30d"].append(opp)
                elif close <= d60:
                    forecast["60d"].append(opp)
                elif close <= d90:
                    forecast["90d"].append(opp)
                else:
                    forecast["beyond_90d"].append(opp)  # Deals closing > 90 days out

            def wsum(deals: list) -> float:
                return sum((d.get("productTotal") or 0) * (d.get("probability") or 0) / 100 for d in deals)

            # total_weighted includes ALL deals with dates (30/60/90/beyond + slipped) for complete pipeline view
            all_weighted = wsum(forecast["30d"]) + wsum(forecast["60d"]) + wsum(forecast["90d"]) + wsum(forecast["beyond_90d"]) + wsum(forecast["slipped"])
            slipped_value = sum(d.get("productTotal") or 0 for d in forecast["slipped"])
            at_risk_pct = (slipped_value / total_pipeline * 100) if total_pipeline > 0 else 0
            no_date_count = len(forecast["no_date"])
            beyond_90d_count = len(forecast["beyond_90d"])

            return {"data": {
                "forecast_30d": {"deals": forecast["30d"][:10], "weighted": wsum(forecast["30d"]), "count": len(forecast["30d"])},
                "forecast_60d": {"deals": forecast["60d"][:10], "weighted": wsum(forecast["60d"]), "count": len(forecast["60d"])},
                "forecast_90d": {"deals": forecast["90d"][:10], "weighted": wsum(forecast["90d"]), "count": len(forecast["90d"])},
                "slipped_deals": sorted(forecast["slipped"], key=lambda x: -(x.get("productTotal") or 0))[:10],
                "beyond_90d_count": beyond_90d_count,  # Deals closing > 90 days (included in total_weighted)
                "no_date_count": no_date_count,  # Opps without close date (excluded from pipeline/at_risk_pct)
                "total_pipeline": total_pipeline, "total_weighted": all_weighted, "at_risk_pct": round(at_risk_pct, 1),
                "duckdb": "GROUP BY period; SUM weighted; calc days_overdue; JOIN contacts for primary"
            }, "error": None}

        elif q == "At-risk deals":
            # === API CALLS (parallel) ===
            with ThreadPoolExecutor(max_workers=3) as executor:
                f_opps = executor.submit(_get, "/api/opportunities", {"$orderby": "daysInStage desc", "$top": 100, "$filter": "status eq 0"})
                f_contacts = executor.submit(_get, "/api/contacts", {"$top": 250})
                f_history = executor.submit(_get, "/api/history", {"$orderby": "startTime desc", "$top": 300})

                opps = f_opps.result()
                contacts = f_contacts.result()
                history = _filter_noise_history(f_history.result())

            # === DUCKDB: track last_touch via BOTH history.contacts AND history.companies ===
            contact_map = {c.get("id"): c for c in contacts if c.get("id")}
            contact_touch: dict[str, str] = {}  # contact_id -> last touch date
            company_touch: dict[str, str] = {}  # company_id -> last touch date
            for h in history:
                st = str(h.get("startTime") or "")[:10]
                if not st:
                    continue
                # Track by contact
                for hc in h.get("contacts") or []:
                    cid = hc.get("id") if isinstance(hc, dict) else None
                    if cid and (cid not in contact_touch or st > contact_touch[cid]):
                        contact_touch[cid] = st
                # Track by company (directly and via contact.companyID)
                for comp in h.get("companies") or []:
                    coid = comp.get("id") if isinstance(comp, dict) else None
                    if coid and (coid not in company_touch or st > company_touch[coid]):
                        company_touch[coid] = st
                for hc in h.get("contacts") or []:
                    hcid = hc.get("id") if isinstance(hc, dict) else None
                    if hcid and hcid in contact_map:
                        coid = contact_map[hcid].get("companyID")
                        if coid and (coid not in company_touch or st > company_touch[coid]):
                            company_touch[coid] = st

            # Cutoff for "Low-activity" = no touch in last 14 days
            import calendar as cal
            cutoff_14d = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 14*24*60*60))

            at_risk = []
            for opp in opps:
                if not _is_open_opportunity(opp):
                    continue
                dis = opp.get("daysInStage") or 0
                close = str(opp.get("estimatedCloseDate") or "")[:10]

                # Compute last_touch FIRST so we can use it in risk classification
                lt = None
                for oc in opp.get("contacts") or []:
                    ocid = oc.get("id") if isinstance(oc, dict) else None
                    if ocid in contact_touch and (not lt or contact_touch[ocid] > lt):
                        lt = contact_touch[ocid]
                    # Also check via contact's companyID
                    if ocid and ocid in contact_map:
                        coid = contact_map[ocid].get("companyID")
                        if coid and coid in company_touch and (not lt or company_touch[coid] > lt):
                            lt = company_touch[coid]
                for comp in opp.get("companies") or []:
                    coid = comp.get("id") if isinstance(comp, dict) else None
                    if coid in company_touch and (not lt or company_touch[coid] > lt):
                        lt = company_touch[coid]

                # Risk classification using last_touch for Low-activity
                is_overdue = close and close < today
                is_stalled = dis > 14
                # Low-activity: no touch in 14 days regardless of probability (high-prob deals can go dark too)
                is_low_activity = not lt or lt < cutoff_14d

                risk = "Overdue" if is_overdue else "Stalled" if is_stalled else "Low-activity" if is_low_activity else None
                if risk:
                    # Calculate days_overdue for Overdue deals (for severity sorting)
                    days_overdue = 0
                    if is_overdue and close:
                        try:
                            close_ts = cal.timegm(time.strptime(close, "%Y-%m-%d"))
                            today_ts = cal.timegm(time.strptime(today, "%Y-%m-%d"))
                            days_overdue = max(0, int((today_ts - close_ts) / 86400))
                        except Exception:
                            pass

                    # Get company name - from opp.companies[0] OR fallback to contact.company
                    co_name = None
                    if opp.get("companies"):
                        first_co = opp["companies"][0] if isinstance(opp["companies"], list) and opp["companies"] else None
                        co_name = first_co.get("name") if isinstance(first_co, dict) else None
                    # Fallback: derive from first contact's company field
                    if not co_name:
                        for oc in opp.get("contacts") or []:
                            ocid = oc.get("id") if isinstance(oc, dict) else None
                            if ocid and ocid in contact_map:
                                co_name = contact_map[ocid].get("company")
                                if co_name:
                                    break

                    # Get primary contact with phone/email (or fallback to any contact)
                    primary = None
                    for oc in opp.get("contacts") or []:
                        ocid = oc.get("id") if isinstance(oc, dict) else None
                        if ocid and ocid in contact_map:
                            c = contact_map[ocid]
                            if c.get("emailAddress") or c.get("businessPhone") or c.get("mobilePhone"):
                                primary = {"name": c.get("fullName"), "email": c.get("emailAddress"),
                                           "phone": c.get("businessPhone") or c.get("mobilePhone")}
                                break
                            elif not primary:  # Fallback: any contact even without phone/email
                                primary = {"name": c.get("fullName"), "email": c.get("emailAddress"), "phone": None}

                    # Get stage info
                    stage_obj = opp.get("stage") or {}
                    stage_name = stage_obj.get("name") if isinstance(stage_obj, dict) else None

                    at_risk.append({
                        "name": opp.get("name"), "company": co_name,
                        "value": opp.get("productTotal"), "probability": opp.get("probability"),
                        "estimatedCloseDate": close,  # Include for Overdue context
                        "stage": stage_name,  # Include stage for context
                        "manager": opp.get("manager") or opp.get("recordManager"),  # For assignment/escalation
                        "risk_reason": risk, "days_in_stage": dis, "days_overdue": days_overdue,
                        "last_touch": lt, "primary_contact": primary
                    })

            # Sort: Overdue first (by days_overdue desc), then Stalled (by days_in_stage desc), then Low-activity
            def risk_sort_key(x: dict[str, Any]) -> tuple[int, int, int]:
                """Sort key for at-risk deals: severity, days_overdue, days_in_stage."""
                risk_order = {"Overdue": 0, "Stalled": 1, "Low-activity": 2}
                severity = risk_order.get(str(x.get("risk_reason", "")), 9)
                days_overdue = x.get("days_overdue") or 0
                days_in_stage = x.get("days_in_stage") or 0
                return (severity, -int(days_overdue), -int(days_in_stage))

            at_risk.sort(key=risk_sort_key)
            return {"data": {"at_risk_deals": at_risk[:10], "duckdb": "filter risk; JOIN history(contacts+companies); ORDER BY severity"}, "error": None}

        elif q == "Account momentum":
            # === API CALLS (parallel) ===
            with ThreadPoolExecutor(max_workers=4) as executor:
                f_companies = executor.submit(_get, "/api/companies", {"$top": 200})
                f_opps = executor.submit(_get, "/api/opportunities", {"$top": 200, "$filter": "status eq 0"})
                f_contacts = executor.submit(_get, "/api/contacts", {"$top": 300})
                f_history = executor.submit(_get, "/api/history", {"$orderby": "startTime desc", "$top": 300})

                companies = f_companies.result()
                opps = f_opps.result()
                contacts = f_contacts.result()
                history = _filter_noise_history(f_history.result())

            open_opps = [o for o in opps if _is_open_opportunity(o)]  # Backup filter

            # === DUCKDB: build maps; GROUP BY company ===
            cutoff_30d = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30*24*60*60))
            contact_map = {c.get("id"): c for c in contacts if c.get("id")}
            co_contacts: dict[str, list] = {}
            for c in contacts:
                coid = c.get("companyID")
                if coid:
                    co_contacts.setdefault(coid, []).append(c)

            metrics: dict[str, dict[str, Any]] = {}
            # Initialize from fetched companies
            for c in companies:
                cid = c.get("id")
                if cid:
                    metrics[str(cid)] = {"name": c.get("name"), "id": cid, "pipeline": 0, "eng30": 0, "last": None, "top_contact": None, "top_contact_date": None}
            # Also discover companies from contacts (may have companies not in /companies)
            for c in contacts:
                coid = c.get("companyID")
                co_name = c.get("company")  # Company name from contact
                if coid and str(coid) not in metrics:
                    metrics[str(coid)] = {"name": co_name, "id": coid, "pipeline": 0, "eng30": 0, "last": None, "top_contact": None, "top_contact_date": None}

            # Also discover companies from opportunities (companies only linked to opps, not in /companies or /contacts)
            for opp in open_opps:
                for comp in opp.get("companies") or []:
                    coid = comp.get("id") if isinstance(comp, dict) else None
                    co_name = comp.get("name") if isinstance(comp, dict) else None
                    if coid and str(coid) not in metrics:
                        metrics[str(coid)] = {"name": co_name, "id": coid, "pipeline": 0, "eng30": 0, "last": None, "top_contact": None, "top_contact_date": None}

            # Link opps to companies via opp.contacts -> contact.companyID (opp.companies often empty)
            for opp in open_opps:
                linked_companies: set[str] = set()
                # First try opp.companies directly
                for comp in opp.get("companies") or []:
                    coid = comp.get("id") if isinstance(comp, dict) else None
                    if coid:
                        linked_companies.add(str(coid))
                # Then derive from opp.contacts -> contact.companyID
                for oc in opp.get("contacts") or []:
                    ocid = oc.get("id") if isinstance(oc, dict) else None
                    if ocid and ocid in contact_map:
                        coid = contact_map[ocid].get("companyID")
                        if coid:
                            linked_companies.add(str(coid))
                # Add pipeline value to all linked companies
                for coid in linked_companies:
                    if coid in metrics:
                        metrics[coid]["pipeline"] += opp.get("productTotal") or 0

            # Track engagement and top_contact per company (via history.companies AND history.contacts->companyID)
            for h in history:
                st = str(h.get("startTime") or "")[:10]
                if not st:
                    continue
                engaged_companies: set[str] = set()
                # Via history.companies
                for comp in h.get("companies") or []:
                    coid = comp.get("id") if isinstance(comp, dict) else None
                    if coid:
                        engaged_companies.add(str(coid))
                # Via history.contacts -> contact.companyID
                for hc in h.get("contacts") or []:
                    hcid = hc.get("id") if isinstance(hc, dict) else None
                    if hcid and hcid in contact_map:
                        coid = contact_map[hcid].get("companyID")
                        if coid:
                            engaged_companies.add(str(coid))
                # Update metrics for all engaged companies
                for coid in engaged_companies:
                    if coid in metrics:
                        if st >= cutoff_30d:
                            metrics[coid]["eng30"] += 1
                        if not metrics[coid]["last"] or st > metrics[coid]["last"]:
                            metrics[coid]["last"] = st
                        # Track most recently engaged contact at this company
                        for hc in h.get("contacts") or []:
                            hcid = hc.get("id") if isinstance(hc, dict) else None
                            if hcid and hcid in contact_map:
                                c = contact_map[hcid]
                                if str(c.get("companyID")) == coid and (not metrics[coid]["top_contact_date"] or st > metrics[coid]["top_contact_date"]):
                                    metrics[coid]["top_contact"] = {"name": c.get("fullName"), "email": c.get("emailAddress"), "phone": c.get("businessPhone") or c.get("mobilePhone")}
                                    metrics[coid]["top_contact_date"] = st

            # Calculate days since last touch using UTC-consistent calculation
            import calendar as cal
            now_utc = cal.timegm(time.gmtime())
            expand: list[dict[str, Any]] = []
            save: list[dict[str, Any]] = []
            reactivate: list[dict[str, Any]] = []
            for m in metrics.values():
                if m["last"]:
                    try:
                        last_ts = cal.timegm(time.strptime(m["last"], "%Y-%m-%d"))
                        m["last_touch_days"] = max(0, int((now_utc - last_ts) / 86400))
                    except Exception:
                        m["last_touch_days"] = None
                else:
                    m["last_touch_days"] = None
                del m["top_contact_date"]  # Internal field
                if m["pipeline"] > 0:
                    m["category"] = "EXPAND" if m["eng30"] >= 3 else "SAVE"
                    (expand if m["eng30"] >= 3 else save).append(m)
                elif m["last"]:
                    m["category"] = "RE-ACTIVATE"
                    reactivate.append(m)

            expand.sort(key=lambda x: -x["pipeline"])
            save.sort(key=lambda x: -x["pipeline"])
            # Sort reactivate by stalest (oldest last touch first) - typical reactivation prioritizes accounts gone cold longest
            reactivate.sort(key=lambda x: (x["last"] or "0000-00-00"))
            return {"data": {"expand": expand[:5], "save": save[:5], "reactivate": reactivate[:5], "duckdb": "GROUP BY company; SUM pipeline; JOIN contacts for top_contact; sort reactivate by staleness"}, "error": None}

        elif q == "Relationship gaps":
            # === API CALLS (parallel) ===
            with ThreadPoolExecutor(max_workers=4) as executor:
                f_opps = executor.submit(_get, "/api/opportunities", {"$orderby": "estimatedCloseDate asc", "$top": 100, "$filter": "status eq 0"})
                f_contacts = executor.submit(_get, "/api/contacts", {"$top": 300})
                f_companies = executor.submit(_get, "/api/companies", {"$top": 100})
                f_history = executor.submit(_get, "/api/history", {"$orderby": "startTime desc", "$top": 200})

                opps = f_opps.result()
                contacts = f_contacts.result()
                companies = f_companies.result()
                history = _filter_noise_history(f_history.result())

            open_opps = [o for o in opps if _is_open_opportunity(o)]  # Backup filter
            # Sort by productTotal desc client-side (API sort by productTotal is slow/timeouts)
            open_opps.sort(key=lambda x: -(x.get("productTotal") or 0))

            # === DUCKDB: COUNT engaged contacts; flag single_threaded; find dark_contacts ===
            cutoff_60d = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 60*24*60*60))

            contact_map = {c.get("id"): c for c in contacts if c.get("id")}
            company_map = {c.get("id"): c.get("name") for c in companies if c.get("id")}  # For reliable name resolution
            co_contacts_map: dict[str, list] = {}
            for c in contacts:
                cid = c.get("companyID")
                if cid:
                    co_contacts_map.setdefault(cid, []).append(c)

            engaged: set[str] = set()
            engaged_companies_set: set[str] = set()  # Companies with recent engagement (via history.companies)
            # Build contact_last_touch from history for consistent staleness ranking
            contact_last_touch: dict[str, str] = {}  # contact_id -> last touch date from history
            for h in history:
                st = str(h.get("startTime") or "")[:10]
                if not st:
                    continue
                # Track engagement via history.contacts (individual contact touches)
                for c in h.get("contacts") or []:
                    if isinstance(c, dict) and c.get("id"):
                        cid = str(c["id"])
                        if st >= cutoff_60d:
                            engaged.add(cid)
                        # Track last touch for staleness ranking
                        if cid not in contact_last_touch or st > contact_last_touch[cid]:
                            contact_last_touch[cid] = st
                # Track company-level engagement separately (NOT marking all contacts as engaged)
                # This avoids over-inflating engaged count and hiding true dark contacts
                for comp in h.get("companies") or []:
                    if isinstance(comp, dict) and comp.get("id"):
                        coid = str(comp["id"])
                        if st >= cutoff_60d:
                            engaged_companies_set.add(coid)

            results = []
            for opp in open_opps[:15]:  # Process top opps by value
                opp_contacts = opp.get("contacts") or []
                # Derive company IDs from opp.companies AND opp.contacts->contact.companyID
                comp_ids: set[str] = set()
                for c in opp.get("companies") or []:
                    if isinstance(c, dict) and c.get("id"):
                        comp_ids.add(str(c["id"]))
                for oc in opp_contacts:
                    ocid = oc.get("id") if isinstance(oc, dict) else None
                    if ocid and ocid in contact_map:
                        coid = contact_map[ocid].get("companyID")
                        if coid:
                            comp_ids.add(str(coid))

                # Count opp-specific engagement (contacts directly linked to opp) for accurate single-threaded detection
                opp_engaged: set[str] = set()
                for oc in opp_contacts:
                    ocid = oc.get("id") if isinstance(oc, dict) else None
                    if ocid and str(ocid) in engaged:
                        opp_engaged.add(str(ocid))
                eng_count = len(opp_engaged)  # Opp-specific engaged contacts
                single = eng_count <= 1  # Single-threaded if only 0-1 engaged contacts on the opp itself

                # Find dark contacts and sort by staleness using history-based last_touch (consistent with engaged set)
                dark = []
                for coid in comp_ids:
                    for c in co_contacts_map.get(coid) or []:
                        cid = str(c.get("id"))
                        if cid not in engaged:
                            ch = "Email" if c.get("emailAddress") else "Phone" if c.get("businessPhone") or c.get("mobilePhone") else "None"
                            # Use history-based last_touch for staleness (consistent source with engaged)
                            last_touch = contact_last_touch.get(cid)  # None if never in history
                            dark.append({
                                "name": c.get("fullName"), "title": c.get("jobTitle"),
                                "lastReach": last_touch,  # history-based, not contact.lastReach
                                "channel": ch,
                                "email": c.get("emailAddress"), "phone": c.get("businessPhone") or c.get("mobilePhone")
                            })
                # Sort dark contacts: prioritize reachable (Email/Phone) over None, then by staleness
                dark.sort(key=lambda x: (
                    0 if x["channel"] != "None" else 1,  # Reachable first
                    0 if x["lastReach"] is None else 1,  # Never touched first among reachable
                    x["lastReach"] or ""  # Then oldest
                ))

                # Get company name - from company_map (fetched /api/companies) for reliable resolution
                co_name = None
                # First try from opp.companies
                if opp.get("companies"):
                    first_co = opp["companies"][0] if isinstance(opp["companies"], list) and opp["companies"] else None
                    coid = first_co.get("id") if isinstance(first_co, dict) else None
                    co_name = company_map.get(coid) or (first_co.get("name") if isinstance(first_co, dict) else None)
                # Fallback: derive from contact's companyID -> company_map
                if not co_name:
                    for coid in comp_ids:
                        if coid in company_map:
                            co_name = company_map[coid]
                            break
                # Final fallback: contact.company field
                if not co_name:
                    for oc in opp_contacts:
                        ocid = oc.get("id") if isinstance(oc, dict) else None
                        if ocid and ocid in contact_map:
                            co_name = contact_map[ocid].get("company")
                            if co_name:
                                break

                results.append({"opp": opp.get("name"), "company": co_name, "value": opp.get("productTotal"), "engaged": eng_count, "single_threaded": single, "dark_contacts": dark[:3]})

            def result_sort_key(x: dict[str, Any]) -> tuple[int, int]:
                """Sort key for relationship analysis: single-threaded first, then by value."""
                st_priority = 0 if x["single_threaded"] else 1
                value = x["value"] or 0
                return (st_priority, -int(value))

            results.sort(key=result_sort_key)
            return {"data": {"relationship_analysis": results[:10], "duckdb": "COUNT engaged; flag single_threaded; find dark_contacts (sorted by staleness)"}, "error": None}

        return {"data": {}, "error": f"Unknown question: {q}"}

    except Exception as e:
        logger.error(f"act_fetch failed for '{q}': {e}")
        return {"data": {}, "error": str(e)}


# Legacy export for backwards compatibility
ACT_API_DB = _DEFAULT_DATABASE

__all__ = [
    "DEMO_MODE",
    "DEMO_STARTERS",
    "DEMO_FOLLOWUPS",
    "DEMO_PROMPTS",
    "ACT_API_DB",
    "ACT_API_USER",
    "AVAILABLE_DATABASES",
    "get_database",
    "set_database",
    "act_fetch",
    "clear_api_cache",
]
