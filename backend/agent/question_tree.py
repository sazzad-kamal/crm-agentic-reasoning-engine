"""
Hardcoded question tree for demo reliability.

This module provides a deterministic tree of questions and follow-ups,
ensuring 100% reliability for demos - no LLM generation variability.

Structure:
- 3 role-based starter questions:
  * Sales Rep (jsmith): "How's my pipeline?" - Pipeline focus
  * CSM (amartin): "Any renewals at risk?" - Retention focus
  * Manager (all): "How's the team doing?" - Aggregate view
- Each leads to 3 follow-ups (layer 2)
- Each leads to 3 follow-ups (layer 3)
- Each leads to 3 follow-ups (layer 4)
- Total: 81 paths (3 * 3 * 3 * 3)

Owner mapping:
- jsmith: Sales Rep (owns ACME-MFG, CROWN-FOODS, FUSION-RETAIL)
- amartin: CSM (owns DELTA-HEALTH)
- ljones: Sales Rep (owns BETA-TECH, GREEN-ENERGY)
- mmalik: Sales Rep (owns EASTERN-TRAVEL, HARBOR-LOGISTICS)

Contact names verified against seed data:
- ACME: Anna Lopez (Ops Manager), Joe Smith (IT), Beth Turner (CFO)
- BETA: Lisa Ng (Head of Sales), Omar Haddad (CEO), Sam Clarke (Sales Ops)
- CROWN: Maria Silva (VP Ops), Alex Dupont (IT Director), Nina Foster (Analyst)
- DELTA: Erin Cho (Medical Director), Mike Tran (IT Manager)

Usage:
    from backend.agent.question_tree import get_follow_ups, get_starters, generate_all_paths

    # Get starter questions
    starters = get_starters()

    # Get follow-ups for a question
    follow_ups = get_follow_ups("How's my pipeline?")

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
    # ROLE-BASED STARTER 1: Sales Rep Pipeline (jsmith)
    # =========================================================================
    "How's my pipeline?": {
        "company_id": None,  # Filtered by owner=jsmith
        "follow_ups": [
            "Which deals need attention?",
            "What's going on with Acme Manufacturing?",
            "What's the forecast for this quarter?",
        ],
    },

    # Layer 2 from Sales Rep Starter
    "Which deals need attention?": {
        "company_id": None,
        "follow_ups": [
            "What stage is the upgrade deal in?",
            "What's blocking the Crown Foods renewal?",
            "Who should I follow up with?",
        ],
    },
    "What's the forecast for this quarter?": {
        "company_id": None,
        "follow_ups": [
            "What's the expected close value?",
            "Which deals are most likely to close?",
            "What's at risk in my pipeline?",
        ],
    },

    # Layer 3 from "Which deals need attention?"
    "What's blocking the Crown Foods renewal?": {
        "company_id": "CROWN-FOODS",
        "follow_ups": [
            "When did we last talk to Crown Foods?",
            "Who is Maria Silva?",
            "What's the renewal value?",
        ],
    },
    "Who should I follow up with?": {
        "company_id": None,
        "follow_ups": [
            "What's the next scheduled activity?",
            "Show me overdue activities",
            "What meetings do I have this week?",
        ],
    },

    # Layer 3 from "What's the forecast for this quarter?"
    "What's the expected close value?": {
        "company_id": None,
        "follow_ups": [
            "What's the weighted pipeline?",
            "Which deals are in negotiation?",
            "How many open opportunities do I have?",
        ],
    },
    "Which deals are most likely to close?": {
        "company_id": None,
        "follow_ups": [
            "Tell me about the Beta Tech negotiation deal",
            "What's the next step for my biggest deal?",
            "Who owns the contacts for these deals?",
        ],
    },
    "What's at risk in my pipeline?": {
        "company_id": None,
        "follow_ups": [
            "Which deals are stalled?",
            "What accounts have low engagement?",
            "Show me deals with no recent activity",
        ],
    },

    # =========================================================================
    # ROLE-BASED STARTER 2: CSM Renewals (amartin)
    # =========================================================================
    "Any renewals at risk?": {
        "company_id": None,  # Filtered by owner=amartin
        "follow_ups": [
            "Tell me about Delta Health's status",
            "Which renewals close this quarter?",
            "What accounts need immediate attention?",
        ],
    },

    # Layer 2 from CSM Starter
    "Tell me about Delta Health's status": {
        "company_id": "DELTA-HEALTH",
        "follow_ups": [
            "When does Delta Health renew?",
            "Who are the contacts at Delta Health?",
            "What's the expansion opportunity?",
        ],
    },
    "Which renewals close this quarter?": {
        "company_id": None,
        "follow_ups": [
            "What's the total renewal value at risk?",
            "Which renewals are in good standing?",
            "Show me renewal timeline",
        ],
    },
    "What accounts need immediate attention?": {
        "company_id": None,
        "follow_ups": [
            "What happened with Green Energy?",
            "Tell me about Eastern Travel's trial",
            "How is Harbor Logistics progressing?",
        ],
    },

    # Layer 3 from "Tell me about Delta Health's status"
    "When does Delta Health renew?": {
        "company_id": "DELTA-HEALTH",
        "follow_ups": [
            "What's the renewal value for Delta Health?",
            "Are there any risks with Delta Health?",
            "What's the engagement level with Delta Health?",
        ],
    },
    "Who are the contacts at Delta Health?": {
        "company_id": "DELTA-HEALTH",
        "follow_ups": [
            "Who is Erin Cho?",
            "What is Mike Tran's role?",
            "When did we last meet with Delta Health?",
        ],
    },
    "What's the expansion opportunity?": {
        "company_id": "DELTA-HEALTH",
        "follow_ups": [
            "What's the expansion deal worth?",
            "Which clinics are we expanding to?",
            "Who's driving the expansion?",
        ],
    },

    # Layer 3 from "Which renewals close this quarter?"
    "What's the total renewal value at risk?": {
        "company_id": None,
        "follow_ups": [
            "Which account has the highest risk?",
            "What's causing the risk?",
            "How can I save these renewals?",
        ],
    },
    "Which renewals are in good standing?": {
        "company_id": None,
        "follow_ups": [
            "What's the total healthy renewal value?",
            "Any upsell opportunities?",
            "When should I reach out?",
        ],
    },
    "Show me renewal timeline": {
        "company_id": None,
        "follow_ups": [
            "What's due next month?",
            "Which renewals need prep?",
            "What's the 90-day renewal forecast?",
        ],
    },

    # =========================================================================
    # ROLE-BASED STARTER 3: Manager Team View (all)
    # =========================================================================
    "How's the team doing?": {
        "company_id": None,  # No owner filter - sees all
        "follow_ups": [
            "What's the total pipeline?",
            "Which reps are behind on activities?",
            "What's the team's forecast?",
        ],
    },

    # Layer 2 from Manager Starter
    "What's the total pipeline?": {
        "company_id": None,
        "follow_ups": [
            "Break down pipeline by rep",
            "What's the average deal size?",
            "Which stage has the most value?",
        ],
    },
    "Which reps are behind on activities?": {
        "company_id": None,
        "follow_ups": [
            "Who has overdue tasks?",
            "What's the activity breakdown by type?",
            "Which accounts need more engagement?",
        ],
    },
    "What's the team's forecast?": {
        "company_id": None,
        "follow_ups": [
            "What's the weighted pipeline by rep?",
            "Tell me about Fusion Retail's upsell",
            "What's at risk this quarter?",
        ],
    },

    # Layer 3 from "What's the total pipeline?"
    "Break down pipeline by rep": {
        "company_id": None,
        "follow_ups": [
            "Who has the biggest deals?",
            "Who needs coaching?",
            "What's the average close rate?",
        ],
    },
    "What's the average deal size?": {
        "company_id": None,
        "follow_ups": [
            "How does this compare to last quarter?",
            "Which segment has the largest deals?",
            "What's the deal size trend?",
        ],
    },
    "Which stage has the most value?": {
        "company_id": None,
        "follow_ups": [
            "Show me deals in negotiation",
            "What's stuck in discovery?",
            "What's the conversion rate by stage?",
        ],
    },

    # Layer 3 from "Which reps are behind on activities?"
    "Who has overdue tasks?": {
        "company_id": None,
        "follow_ups": [
            "What are the overdue tasks?",
            "Which accounts are affected?",
            "When were these tasks due?",
        ],
    },
    "What's the activity breakdown by type?": {
        "company_id": None,
        "follow_ups": [
            "How many calls were made this week?",
            "Which rep has the most meetings?",
            "What's the email activity level?",
        ],
    },
    "Which accounts need more engagement?": {
        "company_id": None,
        "follow_ups": [
            "Show me accounts with no recent activity",
            "What's the average activity per account?",
            "Which accounts are at risk?",
        ],
    },

    # Layer 3 from "What's the team's forecast?"
    "What's the weighted pipeline by rep?": {
        "company_id": None,
        "follow_ups": [
            "Who has the highest weighted value?",
            "Which rep is behind target?",
            "What's the forecast accuracy?",
        ],
    },
    "What's at risk this quarter?": {
        "company_id": None,
        "follow_ups": [
            "Which deals are past due?",
            "What's the total at-risk value?",
            "How can we recover these deals?",
        ],
    },

    # =========================================================================
    # NEW COMPANY PATHS: GREEN-ENERGY, EASTERN-TRAVEL, HARBOR-LOGISTICS, FUSION
    # Addresses missing company coverage from CSM and Manager paths
    # =========================================================================

    # GREEN-ENERGY - Churned account (ljones) - valuable learning case
    "What happened with Green Energy?": {
        "company_id": "GREEN-ENERGY",
        "follow_ups": [
            "Why did Green Energy churn?",
            "Who were the contacts at Green Energy?",
            "What was the timeline of the Green Energy relationship?",
        ],
    },
    "Why did Green Energy churn?": {
        "company_id": "GREEN-ENERGY",
        "follow_ups": [
            "What were the warning signs?",
            "Could we have saved the account?",
            "What can we learn from this?",
        ],
    },
    "Who were the contacts at Green Energy?": {
        "company_id": "GREEN-ENERGY",
        "follow_ups": [
            "Who was Carlos at Green Energy?",
            "What was Linda's role?",
            "When did we last engage with them?",
        ],
    },
    "What was the timeline of the Green Energy relationship?": {
        "company_id": "GREEN-ENERGY",
        "follow_ups": [
            "When did they first become a customer?",
            "What milestones happened?",
            "When did issues start?",
        ],
    },

    # EASTERN-TRAVEL - Trial account (mmalik) - conversion opportunity
    "Tell me about Eastern Travel's trial": {
        "company_id": "EASTERN-TRAVEL",
        "follow_ups": [
            "What's blocking the trial conversion?",
            "Who are the contacts at Eastern Travel?",
            "What's the trial usage like?",
        ],
    },
    "What's blocking the trial conversion?": {
        "company_id": "EASTERN-TRAVEL",
        "follow_ups": [
            "Who has budget authority?",
            "What are their concerns?",
            "When was the trial extended?",
        ],
    },
    "Who are the contacts at Eastern Travel?": {
        "company_id": "EASTERN-TRAVEL",
        "follow_ups": [
            "Who is Sanjay Patel?",
            "What is Ravi's role?",
            "Who should we escalate to?",
        ],
    },
    "What's the trial usage like?": {
        "company_id": "EASTERN-TRAVEL",
        "follow_ups": [
            "Which features are they using?",
            "How active are the users?",
            "What's the engagement trend?",
        ],
    },

    # HARBOR-LOGISTICS - New business (mmalik) - discovery stage
    "How is Harbor Logistics progressing?": {
        "company_id": "HARBOR-LOGISTICS",
        "follow_ups": [
            "What's the status of the Harbor deal?",
            "Who are the contacts at Harbor Logistics?",
            "What's the implementation plan?",
        ],
    },
    "What's the status of the Harbor deal?": {
        "company_id": "HARBOR-LOGISTICS",
        "follow_ups": [
            "What stage is Harbor Logistics in?",
            "What's the deal value?",
            "When is the expected close?",
        ],
    },
    "Who are the contacts at Harbor Logistics?": {
        "company_id": "HARBOR-LOGISTICS",
        "follow_ups": [
            "Who is Jacob Wu?",
            "Who handles operations at Harbor?",
            "Who's the decision maker?",
        ],
    },
    "What's the implementation plan?": {
        "company_id": "HARBOR-LOGISTICS",
        "follow_ups": [
            "What's the rollout timeline?",
            "Which teams are involved?",
            "What training is needed?",
        ],
    },

    # FUSION-RETAIL - Upsell opportunity (jsmith)
    "Tell me about Fusion Retail's upsell": {
        "company_id": "FUSION-RETAIL",
        "follow_ups": [
            "What's the upsell opportunity?",
            "Who are the contacts at Fusion Retail?",
            "What's Fusion's current usage?",
        ],
    },
    "What's the upsell opportunity?": {
        "company_id": "FUSION-RETAIL",
        "follow_ups": [
            "What's the expansion deal worth?",
            "How many additional seats?",
            "What's driving the expansion?",
        ],
    },
    "Who are the contacts at Fusion Retail?": {
        "company_id": "FUSION-RETAIL",
        "follow_ups": [
            "Who is Emma at Fusion?",
            "Who is Lars?",
            "Who approved the original deal?",
        ],
    },
    "What's Fusion's current usage?": {
        "company_id": "FUSION-RETAIL",
        "follow_ups": [
            "How many active users?",
            "Which features are popular?",
            "What's the adoption rate?",
        ],
    },

    # =========================================================================
    # LEGACY STARTER 1: Acme Manufacturing - Activity Focus
    # (Now accessible as follow-up from "How's my pipeline?")
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
            "What documents are attached to Acme deals?",
            "What is Beth Turner's role at Acme?",
        ],
    },

    # Layer 3 from "Show me Acme Manufacturing's opportunities"
    "What stage is the upgrade deal in?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "Who is the contact for the upgrade opportunity?",
            "Show me deals in discovery stage",
            "Which deals have we closed won?",
        ],
    },
    "When does the Acme renewal opportunity close?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What's the expected close date for Acme's renewal?",
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
            "What's the priority of the next Acme activity?",
        ],
    },
    "Show me recent activities with Acme Manufacturing": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What was discussed in the last Acme call?",
            "When was the last meeting with Acme?",
            "How many activities has Acme had this month?",
        ],
    },

    # Layer 3 from "What's Acme Manufacturing's renewal status"
    "When exactly is Acme Manufacturing's renewal date?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What's the health status of Acme Manufacturing?",
            "What proposals have been sent to Acme?",
            "What's the Acme upgrade opportunity worth?",
        ],
    },
    "What documents are attached to Acme deals?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "Is there an expansion opportunity at Acme?",
            "Show me the PDF attachments for Acme",
            "Who is the CFO at Acme Manufacturing?",
        ],
    },
    "What is Beth Turner's role at Acme?": {
        "company_id": "ACME-MFG",
        "follow_ups": [
            "What's Beth Turner's email address?",
            "Has Beth Turner been involved in any deals?",
            "What's the Acme industry segment?",
        ],
    },

    # =========================================================================
    # Beta Tech deal paths (reachable from "Which deals are most likely to close?")
    # =========================================================================
    "Tell me about the Beta Tech negotiation deal": {
        "company_id": "BETA-TECH",
        "follow_ups": [
            "When is the negotiation deal expected to close?",
            "Who is the primary contact for the Beta Tech deal?",
            "What's the negotiation deal value?",
        ],
    },

    # Layer 4 from "Tell me about the Beta Tech negotiation deal"
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
            "Who is Omar Haddad at Beta Tech?",
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

    # Risk identification (reachable from "What's at risk in my pipeline?")
    "Which accounts are at risk?": {
        "company_id": None,
        "follow_ups": [
            "Why is Beta Tech marked at-risk?",
            "Which accounts have the lowest activity?",
            "Show me at-risk renewals",
        ],
    },

    # Activity-related follow-ups (reachable from "Who should I follow up with?")
    "What's the next scheduled activity?": {
        "company_id": None,
        "follow_ups": [
            "Who is the meeting with?",
            "Any activities with trial accounts like Eastern Travel?",
            "What type of activity is it?",
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
# Starter Questions (Entry Points) - Role-Based
# =============================================================================

STARTER_QUESTIONS = [
    "How's my pipeline?",           # Sales Rep (jsmith)
    "Any renewals at risk?",        # CSM (amartin)
    "How's the team doing?",        # Manager (all)
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


