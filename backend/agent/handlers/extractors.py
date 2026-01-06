"""
Extraction helpers for parsing user questions.

Consolidates all pattern matching and parameter extraction
from natural language questions into a single module.
"""


def extract_role_from_question(question: str) -> str | None:
    """Extract contact role from question."""
    question = question.lower()
    if any(word in question for word in ["decision", "maker", "decision-maker"]):
        return "Decision Maker"
    elif "champion" in question:
        return "Champion"
    elif "executive" in question or "vp" in question or "director" in question:
        return "Executive"
    return None


def extract_company_criteria(question: str) -> tuple[str | None, str | None]:
    """Extract segment and industry from question."""
    question = question.lower()
    segment = None
    if "enterprise" in question:
        segment = "Enterprise"
    elif "smb" in question:
        segment = "SMB"
    elif "mid-market" in question or "midmarket" in question:
        segment = "Mid-Market"

    industry = None
    industries = ["software", "manufacturing", "healthcare", "food", "consulting", "retail"]
    for ind in industries:
        if ind in question:
            industry = ind.capitalize()
            break

    return segment, industry


def extract_group_id(question: str) -> str | None:
    """Extract group ID from question keywords."""
    question = question.lower()
    group_keywords = {
        "at risk": "GRP-AT-RISK",
        "at-risk": "GRP-AT-RISK",
        "champion": "GRP-CHAMPIONS",
        "churned": "GRP-CHURNED",
        "dormant": "GRP-DORMANT",
        "hot lead": "GRP-HOT-LEADS",
    }
    for keyword, gid in group_keywords.items():
        if keyword in question:
            return gid
    return None


def extract_attachment_query(question: str) -> str | None:
    """Extract attachment search query from question."""
    question = question.lower()
    search_terms = []
    attachment_words = ["proposal", "contract", "document", "agreement", "pdf", "report"]
    for word in attachment_words:
        if word in question:
            search_terms.append(word)
    return " ".join(search_terms) if search_terms else None


def extract_activity_type(question: str) -> str | None:
    """Extract activity type from question."""
    question = question.lower()
    if "call" in question:
        return "Call"
    elif "email" in question:
        return "Email"
    elif "meeting" in question:
        return "Meeting"
    elif "task" in question:
        return "Task"
    return None


__all__ = [
    "extract_role_from_question",
    "extract_company_criteria",
    "extract_group_id",
    "extract_attachment_query",
    "extract_activity_type",
]
