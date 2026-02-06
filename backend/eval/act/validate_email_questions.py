"""Validate that all 5 email-generation questions are implementable with KQC data.

APPROACH:
1. Filter history by pattern
2. Get contact IDs from matching history (history.contacts[].id)
3. Fetch contact by ID to get email (/api/contacts/{id})
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.act_fetch import _get


def strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"&nbsp;", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


# Cache for contact lookups
_contact_cache: dict[str, dict | None] = {}


def get_contact_by_id(contact_id: str) -> dict | None:
    """Fetch contact by ID to get email (cached)."""
    if contact_id in _contact_cache:
        return _contact_cache[contact_id]
    try:
        result = _get(f"/api/contacts/{contact_id}", {})
        if isinstance(result, list) and result:
            contact = result[0]
        else:
            contact = result
        _contact_cache[contact_id] = contact
        return contact
    except Exception:
        _contact_cache[contact_id] = None
        return None


def validate_questions() -> None:
    """Validate each question using history + contact fetch by ID."""
    print("=" * 80)
    print("VALIDATING EMAIL GENERATION QUESTIONS - KQC Data Check")
    print("=" * 80)

    print("\nFetching history...")
    history = _get("/api/history", {"$top": 1000, "$orderby": "startTime desc"})
    print(f"  History records: {len(history)}")

    with_details = [h for h in history if h.get("details")]
    with_contacts = [h for h in history if h.get("contacts")]
    print(f"  With details: {len(with_details)}")
    print(f"  With contacts linked: {len(with_contacts)}")

    # Define question patterns
    questions = {
        "Q1: Open Quotes": {
            "patterns": [r"quote", r"proposal", r"pricing", r"estimate"],
            "description": "History with quote-related content",
        },
        "Q2: Support Follow-up": {
            "patterns": [r"support", r"help", r"issue", r"problem", r"error", r"fix"],
            "description": "History with support/help content",
        },
        "Q3: Renewal Follow-up": {
            "patterns": [r"renew", r"expir", r"subscript"],
            "description": "History with renewal-related content",
        },
        "Q4: Recent Conversations": {
            "patterns": None,  # Just recent history, no pattern
            "description": "Recent history (any content)",
        },
        "Q5: Technical Issues": {
            "patterns": [r"sync", r"database", r"server", r"install", r"upgrade"],
            "description": "History with technical issue content",
        },
    }

    print("\n" + "=" * 80)

    for q_name, q_config in questions.items():
        print(f"\n### {q_name}")
        print(f"    {q_config['description']}")
        print("-" * 60)

        matches = []
        patterns = q_config["patterns"]
        seen_contact_ids = set()

        # Scan history
        for h in history:
            details = h.get("details")
            contacts_list = h.get("contacts") or []

            if not details or not contacts_list:
                continue

            # Check pattern match
            if patterns is not None:
                details_text = strip_html(details).lower()
                regarding = (h.get("regarding") or "").lower()
                combined = f"{regarding} {details_text}"
                if not any(re.search(p, combined) for p in patterns):
                    continue

            # Get first contact from history
            contact_ref = contacts_list[0]
            contact_id = contact_ref.get("id") if isinstance(contact_ref, dict) else contact_ref
            display_name = contact_ref.get("displayName", "") if isinstance(contact_ref, dict) else ""

            if not contact_id or contact_id in seen_contact_ids:
                continue
            seen_contact_ids.add(contact_id)

            matches.append({
                "history": h,
                "contact_id": contact_id,
                "display_name": display_name,
            })

        print(f"    PATTERN MATCH:")
        print(f"      Matching history records: {len(matches)} (unique contacts)")

        # Fetch emails for first 5 matches
        print(f"\n    EMAIL + TEXT CHECK (fetching contact details):")
        with_email = 0
        for m in matches[:10]:
            contact = get_contact_by_id(m["contact_id"])
            h = m["history"]

            if contact and contact.get("emailAddress"):
                with_email += 1
                email = contact.get("emailAddress")
                name = contact.get("fullName", m["display_name"])
                company = contact.get("company", "")
                details = strip_html(h.get("details", ""))[:120]
                regarding = h.get("regarding", "")[:50]

                if with_email <= 5:  # Show first 5
                    print(f"\n      [{name}] <{email}>")
                    print(f"        Company: {company}")
                    print(f"        Subject: {regarding}")
                    print(f"        Context: {details}...")

        # Verdict
        print(f"\n    VERDICT: ", end="")
        if with_email >= 5:
            print(f"[OK] GOOD - {len(matches)} matches, {with_email}/10 sampled have email")
        elif with_email >= 2:
            print(f"[WARN] LIMITED - {len(matches)} matches, {with_email}/10 have email")
        else:
            print(f"[FAIL] INSUFFICIENT - {len(matches)} matches, {with_email}/10 have email")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
    APPROACH:
    1. Filter history by pattern (quote/support/renewal/tech)
    2. Get contact ID from history.contacts[0].id
    3. Fetch /api/contacts/{id} to get email address

    For implementation:
    - Backend filters history by pattern
    - Fetches contact emails via API
    - AI generates contextual email from history.details
    - Frontend shows mailto: link
    """)


if __name__ == "__main__":
    validate_questions()
