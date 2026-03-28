"""Health Score node - calculates account health metrics."""

import logging
import re
from datetime import datetime
from typing import Any, cast

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import safe_execute
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


HEALTH_WEIGHTS = {
    "deal_value": 0.25,
    "deal_count": 0.15,
    "win_rate": 0.20,
    "activity_recency": 0.15,
    "pipeline_coverage": 0.15,
    "renewal_status": 0.10,
}


def _extract_account_identifier(question: str) -> str | None:
    """Extract account/company name from health score question."""
    patterns = [
        r"health\s+(?:score\s+)?(?:for|of)\s+(.+?)(?:\?|$)",
        r"score\s+(?:for|of)\s+(.+?)(?:\?|$)",
        r"how\s+healthy\s+is\s+(.+?)(?:\?|$)",
        r"(.+?)'?s?\s+(?:health|score|status)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _calculate_deal_score(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate deal-related health metrics."""
    metrics: dict[str, Any] = {
        "total_value": 0,
        "deal_count": 0,
        "won_count": 0,
        "lost_count": 0,
        "open_count": 0,
        "win_rate": 0,
        "avg_deal_size": 0,
    }

    if not data:
        return metrics

    total_value = 0
    won_count = 0
    lost_count = 0
    open_count = 0
    closed_count = 0

    for row in data:
        # Try common column names for deal value
        value = (
            row.get("amount", 0) or
            row.get("value", 0) or
            row.get("deal_value", 0) or
            row.get("revenue", 0) or
            0
        )
        total_value += float(value) if value else 0

        # Check status
        status = str(row.get("status", "") or row.get("stage", "") or "").lower()
        if status in ("won", "closed won", "closed-won", "closed_won"):
            won_count += 1
            closed_count += 1
        elif status in ("lost", "closed lost", "closed-lost", "closed_lost"):
            lost_count += 1
            closed_count += 1
        else:
            open_count += 1

    metrics["total_value"] = total_value
    metrics["deal_count"] = len(data)
    metrics["won_count"] = won_count
    metrics["lost_count"] = lost_count
    metrics["open_count"] = open_count
    metrics["win_rate"] = (won_count / closed_count * 100) if closed_count > 0 else 0
    metrics["avg_deal_size"] = total_value / len(data) if data else 0

    return metrics


def _calculate_activity_score(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate activity recency score."""
    metrics: dict[str, Any] = {
        "last_activity_days": None,
        "activity_count": 0,
        "recency_score": 0,
    }

    if not data:
        return metrics

    metrics["activity_count"] = len(data)

    # Try to find most recent activity date
    date_fields = ["activity_date", "created_at", "updated_at", "date", "last_contact"]
    most_recent = None

    for row in data:
        for field in date_fields:
            date_val = row.get(field)
            if date_val:
                try:
                    if isinstance(date_val, str):
                        date_val = datetime.fromisoformat(date_val.replace("Z", "+00:00"))
                    if most_recent is None or date_val > most_recent:
                        most_recent = date_val
                except (ValueError, TypeError):
                    continue

    if most_recent:
        days_ago = (datetime.now(most_recent.tzinfo) - most_recent).days
        metrics["last_activity_days"] = days_ago
        # Score based on recency (100 if today, 0 if > 90 days)
        metrics["recency_score"] = max(0, 100 - (days_ago * 100 / 90))

    return metrics


def _compute_health_score(
    deal_metrics: dict[str, Any],
    activity_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compute overall health score from component metrics."""
    score_components: dict[str, float] = {}

    # Deal value score (normalized to 0-100)
    # Using logarithmic scale for deal value
    value = deal_metrics.get("total_value", 0)
    if value > 0:
        import math
        # Score from 0-100 based on log scale (10k = 50, 100k = 75, 1M = 100)
        value_score = min(100, 25 * math.log10(max(value, 1)))
    else:
        value_score = 0
    score_components["deal_value"] = value_score

    # Deal count score
    count = deal_metrics.get("deal_count", 0)
    count_score = min(100, count * 10)  # 10 deals = 100
    score_components["deal_count"] = count_score

    # Win rate score
    win_rate = deal_metrics.get("win_rate", 0)
    score_components["win_rate"] = win_rate

    # Activity recency score
    recency = activity_metrics.get("recency_score", 0)
    score_components["activity_recency"] = recency

    # Pipeline coverage (simplified: open deals vs closed)
    open_count = deal_metrics.get("open_count", 0)
    closed = deal_metrics.get("won_count", 0) + deal_metrics.get("lost_count", 0)
    if closed > 0:
        coverage = min(100, open_count / closed * 100)
    else:
        coverage = 100 if open_count > 0 else 0
    score_components["pipeline_coverage"] = coverage

    # Renewal status (placeholder - would need actual renewal data)
    score_components["renewal_status"] = 50  # Neutral

    # Calculate weighted score
    total_score = 0
    for component, weight in HEALTH_WEIGHTS.items():
        total_score += score_components.get(component, 0) * weight

    return {
        "overall_score": round(total_score, 1),
        "grade": _score_to_grade(total_score),
        "components": score_components,
    }


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def _get_health_insights(
    health_score: dict[str, Any],
    deal_metrics: dict[str, Any],
    activity_metrics: dict[str, Any],
) -> list[str]:
    """Generate actionable insights based on health metrics."""
    insights = []
    components = health_score.get("components", {})

    # Check for low scores and suggest improvements
    if components.get("activity_recency", 0) < 50:
        days = activity_metrics.get("last_activity_days")
        if days:
            insights.append(f"No activity in {days} days - schedule a check-in call")
        else:
            insights.append("No recent activity recorded - reach out to maintain relationship")

    if components.get("win_rate", 0) < 30:
        insights.append("Win rate below 30% - review sales approach for this account")

    if components.get("pipeline_coverage", 0) < 30:
        insights.append("Limited active pipeline - explore expansion opportunities")

    if deal_metrics.get("deal_count", 0) < 3:
        insights.append("Few deals on record - potential for deeper engagement")

    if health_score.get("overall_score", 0) >= 80:
        insights.append("Strong account health - consider for case study or referral")

    return insights


def _safe_fetch(query: str, history: str, prefix: str) -> list[dict[str, Any]]:
    """Plan SQL, validate, and execute — returns rows or empty list on failure."""
    try:
        plan = get_sql_plan(question=query, conversation_history=history)
        if plan.sql:
            rows, _ = safe_execute(plan.sql, get_connection())
            return rows
        return []
    except Exception as e:
        logger.error(f"{prefix} fetch failed: {e}")
        return []


def health_node(state: AgentState) -> AgentState:
    """Health Score node that calculates account health metrics."""
    question = state["question"]
    history = format_conversation_for_prompt(state.get("messages", []))
    logger.info(f"[Health] Processing: {question[:50]}...")

    result: dict[str, Any] = {
        "sql_results": {},
    }

    # Step 1: Extract account identifier
    account = _extract_account_identifier(question)

    if not account:
        # If no specific account, analyze all accounts
        account = "all accounts"
        account_filter = ""
    else:
        account_filter = f"for company/account named '{account}'"
        logger.info(f"[Health] Analyzing account: {account}")

    # Step 2: Fetch deal data
    deal_query = f"Get all deals {account_filter} with status, amount, and dates"

    deal_data = _safe_fetch(deal_query, history, "[Health] Deal")

    # Step 3: Fetch activity data
    activity_query = f"Get recent activities {account_filter} with dates"
    activity_data = _safe_fetch(activity_query, history, "[Health] Activity")

    # Step 4: Calculate metrics
    deal_metrics = _calculate_deal_score(deal_data)
    activity_metrics = _calculate_activity_score(activity_data)

    # Step 5: Compute overall health score
    health_score = _compute_health_score(deal_metrics, activity_metrics)

    # Step 6: Generate insights
    insights = _get_health_insights(health_score, deal_metrics, activity_metrics)

    # Build result
    sql_results: dict[str, Any] = {
        "_debug": {
            "account": account,
            "deal_count": len(deal_data),
            "activity_count": len(activity_data),
        },
        "data": deal_data,  # Primary data for downstream
        "health_analysis": {
            "account": account,
            "score": health_score["overall_score"],
            "grade": health_score["grade"],
            "components": health_score["components"],
            "deal_metrics": deal_metrics,
            "activity_metrics": activity_metrics,
            "insights": insights,
        },
    }

    result["sql_results"] = sql_results
    logger.info(f"[Health] Complete: Score {health_score['overall_score']} ({health_score['grade']})")

    return cast(AgentState, result)


__all__ = ["health_node"]
