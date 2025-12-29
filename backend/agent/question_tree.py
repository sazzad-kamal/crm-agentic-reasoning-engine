"""
Hardcoded question tree for demo reliability.

This module provides a deterministic tree of questions and follow-ups,
ensuring 100% reliability for demos - no LLM generation variability.

Structure:
- 5 starter questions (all CRM data focused)
- Each leads to 3 follow-ups (layer 2)
- Each leads to 3 follow-ups (layer 3)
- Each leads to 3 follow-ups (layer 4)
- Total: 135 paths (5 * 3 * 3 * 3)

Contact names verified against seed data:
- ACME: Anna Lopez (Ops Manager), Joe Smith (IT), Beth Turner (CFO)
- BETA: Lisa Ng (Head of Sales), Omar Haddad (CEO), Sam Clarke (Sales Ops)
- CROWN: Maria Silva (VP Ops), Alex Dupont (IT Director), Nina Foster (Analyst)

Usage:
    from backend.agent.question_tree import get_follow_ups, get_starters, generate_all_paths

    # Get starter questions
    starters = get_starters()

    # Get follow-ups for a question
    follow_ups = get_follow_ups("What's going on with Acme Manufacturing?")

    # Generate all paths for testing
    paths = generate_all_paths()
"""

from typing import TypedDict


class QuestionNode(TypedDict):
    """A node in the question tree."""
    company_id: str | None  # Company context (if any)
    follow_ups: list[str]   # Next questions user can ask


# =============================================================================
# Question Tree Definition
# =============================================================================
# Structure: question_text -> {company_id, follow_ups}
# Each question maps to its valid follow-up questions

QUESTION_TREE: dict[str, QuestionNode] = {
    # =========================================================================
    # STARTER 1: Acme Manufacturing - Activity Focus
    # =========================================================================
    "What's going on with Acme Manufacturing?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "Show me Acme Manufacturing's opportunities",
            "Who are the key contacts at Acme Manufacturing?",
            "What's Acme Manufacturing's renewal status?",
        ],
    },

    # Layer 2 from Starter 1
    "Show me Acme Manufacturing's opportunities": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What stage is the upgrade deal in?",
            "When does the Acme renewal opportunity close?",
            "What's the total pipeline value for Acme Manufacturing?",
        ],
    },
    "Who are the key contacts at Acme Manufacturing?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What is Anna Lopez's role?",
            "Who handles IT at Acme Manufacturing?",
            "Show me recent activities with Acme Manufacturing",
        ],
    },
    "What's Acme Manufacturing's renewal status?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "When exactly is Acme Manufacturing's renewal date?",
            "What's the renewal amount for Acme Manufacturing?",
            "Who owns the Acme Manufacturing account?",
        ],
    },

    # Layer 3 from "Show me Acme Manufacturing's opportunities"
    "What stage is the upgrade deal in?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "Who is the contact for the upgrade opportunity?",
            "What's the upgrade deal worth?",
            "How many Acme deals are in Proposal stage?",
        ],
    },
    "When does the Acme renewal opportunity close?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What's the renewal value for Acme?",
            "Who's handling the Acme Manufacturing renewal?",
            "What's the current stage of Acme's renewal?",
        ],
    },
    "What's the total pipeline value for Acme Manufacturing?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What's the biggest opportunity for Acme?",
            "How many opportunities does Acme have?",
            "Who is the account owner for Acme?",
        ],
    },

    # Layer 3 from "Who are the key contacts"
    "What is Anna Lopez's role?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What deals is Anna Lopez involved in?",
            "When did we last talk to Anna Lopez?",
            "What's the breakdown of contact roles at Acme?",
        ],
    },
    "Who handles IT at Acme Manufacturing?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What is Joe Smith's contact info?",
            "What deals involve Joe Smith?",
            "Show me activities with Joe Smith",
        ],
    },
    "Show me recent activities with Acme Manufacturing": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What was the last call about?",
            "When was the last meeting with Acme?",
            "How many activities has Acme had this month?",
        ],
    },

    # Layer 3 from "What's Acme Manufacturing's renewal status"
    "When exactly is Acme Manufacturing's renewal date?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What's the health status of Acme Manufacturing?",
            "Who should I contact about the Acme renewal?",
            "What is the Acme renewal amount?",
        ],
    },
    "What's the renewal amount for Acme Manufacturing?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "Is there an expansion opportunity at Acme?",
            "Show me all Acme Manufacturing opportunities",
            "Who is the primary contact for Acme?",
        ],
    },
    "Who owns the Acme Manufacturing account?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What other accounts does jsmith own?",
            "Show me the Acme account details",
            "What's the Acme industry segment?",
        ],
    },

    # =========================================================================
    # STARTER 2: Beta Tech - Pipeline Focus
    # =========================================================================
    "Show me Beta Tech's pipeline": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "Tell me about the Beta Tech negotiation deal",
            "What's the Beta Tech add-on deal?",
            "Who are the contacts at Beta Tech?",
        ],
    },

    # Layer 2 from Starter 2
    "Tell me about the Beta Tech negotiation deal": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "When is the negotiation deal expected to close?",
            "Who is the primary contact for the Beta Tech deal?",
            "What's the negotiation deal value?",
        ],
    },
    "What's the Beta Tech add-on deal?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What stage is the add-on in?",
            "Who's the contact for the add-on deal?",
            "When does Beta Tech's contract renew?",
        ],
    },
    "Who are the contacts at Beta Tech?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What is Lisa Ng's role?",
            "Tell me about Sam Clarke",
            "Show me activities with Beta Tech",
        ],
    },

    # Layer 3 from "Tell me about the Beta Tech negotiation deal"
    "When is the negotiation deal expected to close?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What's the total value of Beta Tech deals?",
            "Who should I follow up with at Beta Tech?",
            "What activities happened with Beta Tech?",
        ],
    },
    "Who is the primary contact for the Beta Tech deal?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What's Lisa Ng's job title?",
            "When did we last talk to Beta Tech?",
            "What is Lisa Ng's email?",
        ],
    },
    "What's the negotiation deal value?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What's Beta Tech's total pipeline?",
            "How many deals does Beta Tech have?",
            "What's the add-on deal worth?",
        ],
    },

    # Layer 3 from "What's the Beta Tech add-on deal"
    "What stage is the add-on in?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "How many Beta Tech deals are in Discovery stage?",
            "Who's driving the add-on deal?",
            "Show me all Beta Tech opportunities",
        ],
    },
    "Who's the contact for the add-on deal?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What is Sam Clarke's role?",
            "When did we last engage with Sam Clarke?",
            "What other contacts are at Beta Tech?",
        ],
    },
    "When does Beta Tech's contract renew?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What's the Beta Tech renewal value?",
            "Is there expansion potential at Beta Tech?",
            "What's the health status of Beta Tech?",
        ],
    },

    # Layer 3 from "Who are the contacts at Beta Tech"
    "What is Lisa Ng's role?": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What deals is Lisa Ng involved in?",
            "When did we last meet with Lisa Ng?",
            "Who else is at Beta Tech?",
        ],
    },
    "Tell me about Sam Clarke": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What opportunities involve Sam Clarke?",
            "What is Sam Clarke's email?",
            "What's Sam Clarke's job title?",
        ],
    },
    "Show me activities with Beta Tech": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "What was the last Beta Tech meeting about?",
            "Who attended the recent Beta Tech calls?",
            "What's Beta Tech's activity count this month?",
        ],
    },

    # =========================================================================
    # STARTER 3: Renewals - Cross-Company Focus
    # =========================================================================
    "Which renewals are coming up this month?": {
        "company_id": None,  # General query
        "follow_ups": [
            "Tell me more about Crown Foods' renewal",
            "What's the health status of accounts with upcoming renewals?",
            "Which renewal has the highest value?",
        ],
    },

    # Layer 2 from Starter 3
    "Tell me more about Crown Foods' renewal": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "Who is the contact for Crown Foods?",
            "What's the Crown Foods renewal value?",
            "What activities happened with Crown Foods?",
        ],
    },
    "What's the health status of accounts with upcoming renewals?": {
        "company_id": None,
        "follow_ups": [
            "Which accounts are at risk?",
            "How many accounts are in each health status?",
            "Show me accounts with low activity",
        ],
    },
    "Which renewal has the highest value?": {
        "company_id": None,
        "follow_ups": [
            "What's the total renewal value this quarter?",
            "Who owns the highest value renewal?",
            "When do the other renewals close?",
        ],
    },

    # Layer 3 from "Tell me more about Crown Foods' renewal"
    "Who is the contact for Crown Foods?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "What is Maria Silva's role?",
            "When did we last talk to Crown Foods?",
            "Show me Crown Foods' opportunities",
        ],
    },
    "What's the Crown Foods renewal value?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "Is there upsell potential at Crown Foods?",
            "Who owns the Crown Foods account?",
            "What's the next renewal after Crown Foods?",
        ],
    },
    "What activities happened with Crown Foods?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "What was discussed in the last Crown Foods call?",
            "Who attended the Crown Foods meetings?",
            "What's the next scheduled activity with Crown Foods?",
        ],
    },

    # Layer 3 from "What's the health status"
    "Which accounts are at risk?": {
        "company_id": None,
        "follow_ups": [
            "Why is Beta Tech marked at-risk?",
            "Which accounts have the lowest activity?",
            "Show me at-risk renewals",
        ],
    },
    "How many accounts are in each health status?": {
        "company_id": None,
        "follow_ups": [
            "Which accounts have healthy status?",
            "What's the average activity count for at-risk accounts?",
            "When was the last activity for at-risk accounts?",
        ],
    },
    "Show me accounts with low activity": {
        "company_id": None,
        "follow_ups": [
            "Which accounts need immediate attention?",
            "What's the last activity for each account?",
            "What types of activities have been logged recently?",
        ],
    },

    # Layer 3 from "Which renewal has the highest value"
    "What's the total renewal value this quarter?": {
        "company_id": None,
        "follow_ups": [
            "Break down renewals by owner",
            "Which renewals are at risk?",
            "How many accounts churned this quarter?",
        ],
    },
    "Who owns the highest value renewal?": {
        "company_id": None,
        "follow_ups": [
            "What other accounts does this owner have?",
            "Show me the account details",
            "What's the owner's total pipeline?",
        ],
    },
    "When do the other renewals close?": {
        "company_id": None,
        "follow_ups": [
            "Which renewals close next quarter?",
            "Show me renewals by close date",
            "What's the total pipeline for renewals?",
        ],
    },

    # =========================================================================
    # STARTER 4: Portfolio Overview - Global/Aggregate Focus
    # Tests: pipeline_summary, company_search, groups, global activities
    # =========================================================================
    "What's our total pipeline?": {
        "company_id": None,  # Global aggregate query
        "follow_ups": [
            "Which companies have the biggest deals?",
            "What groups do we have?",
            "What meetings do we have coming up?",
        ],
    },

    # Layer 2 from Starter 4
    "Which companies have the biggest deals?": {
        "company_id": None,
        "follow_ups": [
            "Show me all enterprise accounts",
            "What's the pipeline breakdown by owner?",
            "Which deals are in negotiation stage?",
        ],
    },
    "What groups do we have?": {
        "company_id": None,
        "follow_ups": [
            "Who is in the at-risk accounts group?",
            "Show me the churned accounts group",
            "What's the distribution of accounts by group?",
        ],
    },
    "What meetings do we have coming up?": {
        "company_id": None,
        "follow_ups": [
            "Show me all recent calls",
            "What's the next scheduled activity?",
            "What's the breakdown of activity types this month?",
        ],
    },

    # Layer 3 from "Which companies have the biggest deals"
    "Show me all enterprise accounts": {
        "company_id": None,
        "follow_ups": [
            "Tell me about Fusion Retail Group",
            "What's the total enterprise pipeline value?",
            "Who are the decision makers at enterprise accounts?",
        ],
    },
    "What's the pipeline breakdown by owner?": {
        "company_id": None,
        "follow_ups": [
            "Who has the most deals?",
            "What accounts does mmalik own? Like Harbor Logistics?",
            "Show me jsmith's accounts",
        ],
    },
    "Which deals are in negotiation stage?": {
        "company_id": None,
        "follow_ups": [
            "How many negotiation deals do we have?",
            "Which negotiation deals close this month?",
            "Who owns the negotiation deals?",
        ],
    },

    # Layer 3 from "What groups do we have"
    "Who is in the at-risk accounts group?": {
        "company_id": None,
        "follow_ups": [
            "Tell me about Delta Health's risk status",
            "Show me the last activity for at-risk accounts",
            "What's the pipeline for at-risk accounts?",
        ],
    },
    "Show me the churned accounts group": {
        "company_id": None,
        "follow_ups": [
            "What happened with Green Energy Partners?",
            "When did these accounts churn?",
            "What was the total value of churned accounts?",
        ],
    },
    "What's the distribution of accounts by group?": {
        "company_id": None,
        "follow_ups": [
            "How many accounts are in each group?",
            "What's the pipeline value for at-risk accounts?",
            "Which group has the highest total value?",
        ],
    },

    # Layer 3 from "What meetings do we have coming up"
    "Show me all recent calls": {
        "company_id": None,
        "follow_ups": [
            "Which accounts had calls this week?",
            "Who made the most calls?",
            "What was discussed in the last call?",
        ],
    },
    "What's the next scheduled activity?": {
        "company_id": None,
        "follow_ups": [
            "Who is the meeting with?",
            "Any activities with trial accounts like Eastern Travel?",
            "What type of activity is it?",
        ],
    },
    "What's the breakdown of activity types this month?": {
        "company_id": None,
        "follow_ups": [
            "What percentage are calls vs meetings?",
            "How many emails were logged this month?",
            "Which activity type is most common?",
        ],
    },

    # =========================================================================
    # STARTER 5: Account Deep Dive - Crown Foods
    # Tests: account_context, complex summaries, attachments, notes search
    # =========================================================================
    "Give me the full picture on Crown Foods": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "Who are the key contacts at Crown Foods?",
            "What's the status of Crown Foods' renewal?",
            "Show me all Crown Foods activities",
        ],
    },

    # Layer 2 from Starter 5
    "Who are the key contacts at Crown Foods?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "What is Maria Silva's role?",
            "Who handles IT at Crown Foods?",
            "Show me the decision makers at Crown Foods",
        ],
    },
    "What's the status of Crown Foods' renewal?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "When exactly does Crown Foods renew?",
            "What's the renewal value for Crown Foods?",
            "Are there any risks with the Crown Foods renewal?",
        ],
    },
    "Show me all Crown Foods activities": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "What was the last call with Crown Foods about?",
            "Who attended the recent Crown Foods meetings?",
            "What's the next scheduled activity with Crown Foods?",
        ],
    },

    # Layer 3 from "Who are the key contacts at Crown Foods"
    "What is Maria Silva's role?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "What deals is Maria Silva involved in?",
            "When did we last talk to Maria Silva?",
            "What's Maria Silva's contact info?",
        ],
    },
    "Who handles IT at Crown Foods?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "What is Alex Dupont's role?",
            "Show me activities with Alex Dupont",
            "What deals involve Alex Dupont?",
        ],
    },
    "Show me the decision makers at Crown Foods": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "Who should I contact about the renewal?",
            "What's the VP's email address?",
            "Who else influences decisions at Crown Foods?",
        ],
    },

    # Layer 3 from "What's the status of Crown Foods' renewal"
    "When exactly does Crown Foods renew?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "How many days until Crown Foods renews?",
            "What's the Crown Foods account health?",
            "Who owns the Crown Foods account?",
        ],
    },
    "What's the renewal value for Crown Foods?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "Is there upsell potential at Crown Foods?",
            "What plan is Crown Foods on?",
            "Compare Crown Foods to similar accounts",
        ],
    },
    "Are there any risks with the Crown Foods renewal?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "What's the engagement level with Crown Foods?",
            "Have there been any complaints from Crown Foods?",
            "What's Crown Foods' usage like?",
        ],
    },

    # Layer 3 from "Show me all Crown Foods activities"
    "What was the last call with Crown Foods about?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "Who was on that call?",
            "What action items came out of it?",
            "When is the next follow-up scheduled?",
        ],
    },
    "Who attended the recent Crown Foods meetings?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "How many meetings have we had with Crown Foods?",
            "What topics were discussed?",
            "Who from our team attended?",
        ],
    },
    "What's the next scheduled activity with Crown Foods?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "What type of activity is it?",
            "Who is attending?",
            "Who are the attendees for this activity?",
        ],
    },

    # =========================================================================
    # Layer 4 Terminal Questions (these have generic follow-ups)
    # =========================================================================
}

# Terminal nodes (layer 4) return empty list - no more suggested buttons
# This prevents infinite loops and signals end of guided demo flow
TERMINAL_FOLLOW_UPS: list[str] = []


# =============================================================================
# Starter Questions (Entry Points)
# =============================================================================

STARTER_QUESTIONS = [
    "What's going on with Acme Manufacturing?",
    "Show me Beta Tech's pipeline",
    "Which renewals are coming up this month?",
    "What's our total pipeline?",
    "Give me the full picture on Crown Foods",
]


# =============================================================================
# Public API
# =============================================================================

def get_starters() -> list[str]:
    """Get the starter questions."""
    return STARTER_QUESTIONS.copy()


def get_follow_ups(question: str) -> list[str]:
    """
    Get follow-up questions for a given question.

    Returns hardcoded follow-ups from the tree, or terminal follow-ups
    if the question isn't in the tree.
    """
    node = QUESTION_TREE.get(question)
    if node:
        return node["follow_ups"].copy()
    return TERMINAL_FOLLOW_UPS.copy()


def get_company_id(question: str) -> str | None:
    """Get the company_id context for a question."""
    node = QUESTION_TREE.get(question)
    if node:
        return node.get("company_id")
    return None


def generate_all_paths(max_depth: int = 4) -> list[list[str]]:
    """
    Generate all possible conversation paths through the tree.

    Args:
        max_depth: Maximum number of questions in a path (default 4)

    Returns:
        List of paths, where each path is a list of questions.
        With 3 starters and 3 follow-ups at each level:
        - depth=2: 9 paths (3 * 3)
        - depth=3: 27 paths (3 * 3 * 3)
        - depth=4: 81 paths (3 * 3 * 3 * 3)
    """
    paths: list[list[str]] = []

    def _traverse(current_path: list[str], depth: int):
        if depth >= max_depth:
            paths.append(current_path.copy())
            return

        current_question = current_path[-1] if current_path else None

        if current_question is None:
            # Start with starters
            for starter in STARTER_QUESTIONS:
                _traverse([starter], depth + 1)
        else:
            # Get follow-ups for current question
            follow_ups = get_follow_ups(current_question)
            if not follow_ups or follow_ups == TERMINAL_FOLLOW_UPS:
                # Terminal node - end this path
                paths.append(current_path.copy())
                return

            for follow_up in follow_ups:
                _traverse(current_path + [follow_up], depth + 1)

    _traverse([], 0)
    return paths


def get_tree_stats() -> dict:
    """Get statistics about the question tree."""
    paths = generate_all_paths()
    return {
        "num_starters": len(STARTER_QUESTIONS),
        "num_questions": len(QUESTION_TREE),
        "num_paths": len(paths),
        "path_lengths": {
            "min": min(len(p) for p in paths) if paths else 0,
            "max": max(len(p) for p in paths) if paths else 0,
        },
    }


# =============================================================================
# Validation
# =============================================================================

def validate_tree() -> list[str]:
    """
    Validate the question tree for consistency.

    Returns list of any issues found.
    """
    issues = []

    # Check all starters are in tree
    for starter in STARTER_QUESTIONS:
        if starter not in QUESTION_TREE:
            issues.append(f"Starter not in tree: {starter}")

    # Check all follow-ups reference valid questions or are terminal
    for question, node in QUESTION_TREE.items():
        for follow_up in node["follow_ups"]:
            if follow_up not in QUESTION_TREE and follow_up not in TERMINAL_FOLLOW_UPS:
                # This is a layer 4 question - should use terminal follow-ups
                pass  # OK, will get terminal follow-ups

    # Check for orphaned questions (not reachable from starters)
    reachable = set()

    def _mark_reachable(q: str):
        if q in reachable:
            return
        reachable.add(q)
        node = QUESTION_TREE.get(q)
        if node:
            for follow_up in node["follow_ups"]:
                _mark_reachable(follow_up)

    for starter in STARTER_QUESTIONS:
        _mark_reachable(starter)

    for question in QUESTION_TREE:
        if question not in reachable:
            issues.append(f"Orphaned question (not reachable): {question}")

    return issues


if __name__ == "__main__":
    # Quick test
    print("Question Tree Stats:")
    stats = get_tree_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nValidation:")
    issues = validate_tree()
    if issues:
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print("  All checks passed!")

    print("\nSample paths:")
    paths = generate_all_paths()
    for i, path in enumerate(paths[:3]):
        print(f"\n  Path {i+1}:")
        for j, q in enumerate(path):
            print(f"    {j+1}. {q}")
