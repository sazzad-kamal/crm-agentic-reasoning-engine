"""CRM data fetch node for LangGraph parallel execution."""

import logging
import time

from backend.agent.core.state import AgentState
from backend.agent.core.config import get_config
from backend.agent.fetch.handlers import IntentContext, dispatch_intent


logger = logging.getLogger(__name__)


def fetch_crm_node(state: AgentState) -> AgentState:
    """Fetch CRM data based on intent."""
    config = get_config()
    start_time = time.time()

    question = state.get("question", "")
    intent = state.get("intent", "general")
    company_id = state.get("resolved_company_id")
    days = state.get("days", config.default_days)
    company_name_query = state.get("company_name_query")
    owner = state.get("owner")

    logger.info(f"[FetchCRM] Fetching data for intent={intent}, company={company_id}")

    try:
        ctx = IntentContext(
            question=question.lower(),
            resolved_company_id=company_id,
            days=days,
            company_name_query=company_name_query,
            owner=owner,
        )
        result = dispatch_intent(intent, ctx)

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[FetchCRM] Complete in {latency_ms}ms, sources={len(result.sources)}")

        return {
            "company_data": result.company_data,
            "activities_data": result.activities_data,
            "history_data": result.history_data,
            "pipeline_data": result.pipeline_data,
            "renewals_data": result.renewals_data,
            "contacts_data": result.contacts_data,
            "groups_data": result.groups_data,
            "attachments_data": result.attachments_data,
            "resolved_company_id": result.resolved_company_id or ctx.resolved_company_id,
            "sources": result.sources,
            "raw_data": result.raw_data,
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[FetchCRM] Failed after {latency_ms}ms: {e}")
        return {"error": f"CRM fetch failed: {e}"}


__all__ = ["fetch_crm_node"]
