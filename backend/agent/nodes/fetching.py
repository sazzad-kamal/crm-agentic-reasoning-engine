"""
LangGraph fetch node for agent workflow.

Handles parallel data fetching from CRM, docs, and account context.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from backend.agent.state import AgentState
from backend.agent.config import get_config
from backend.agent.llm_helpers import call_docs_rag, call_account_rag
from backend.agent.handlers import IntentContext, dispatch_intent


logger = logging.getLogger(__name__)


# Intents that trigger Account RAG for unstructured text search
ACCOUNT_RAG_INTENTS = frozenset(
    {
        "account_context",
        "company_status",
        "history",
        "pipeline",
    }
)


# =============================================================================
# Fetch Helper Functions (module-level for testability)
# =============================================================================


def _fetch_crm_data(
    question: str,
    intent: str,
    resolved_company_id: str | None,
    days: int,
    router_result: object | None,
    owner: str | None = None,
) -> dict:
    """
    Fetch CRM data based on intent.

    Extracted to module level for testability.
    """
    try:
        ctx = IntentContext(
            question=question.lower(),
            resolved_company_id=resolved_company_id,
            days=days,
            router_result=router_result,
            owner=owner,
        )
        result = dispatch_intent(intent, ctx)
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
        logger.error(f"[Fetch] Data fetch failed: {e}")
        return {"error": str(e)}


def _fetch_docs(question: str) -> dict:
    """
    Fetch documentation via RAG.

    Extracted to module level for testability.
    """
    try:
        docs_answer, docs_sources = call_docs_rag(question)
        return {
            "docs_answer": docs_answer,
            "docs_sources": docs_sources,
        }
    except Exception as e:
        logger.error(f"[Fetch] Docs fetch failed: {e}")
        return {"docs_answer": "", "docs_sources": [], "error": str(e)}


def _fetch_account_context(question: str, company_id: str) -> dict:
    """
    Fetch account context via Account RAG.

    Extracted to module level for testability.
    """
    try:
        account_answer, account_sources = call_account_rag(
            question=question,
            company_id=company_id,
        )
        return {
            "account_context_answer": account_answer,
            "account_context_sources": account_sources,
        }
    except Exception as e:
        logger.error(f"[Fetch] Account RAG failed: {e}")
        return {"account_context_answer": "", "account_context_sources": [], "error": str(e)}


# =============================================================================
# Fetch Node
# =============================================================================


def fetch_node(state: AgentState) -> AgentState:
    """
    Unified fetch node: Fetch CRM data, docs, and account context in parallel.

    Always fetches both CRM data and documentation concurrently using
    ThreadPoolExecutor. For company-specific intents, also fetches
    Account RAG (notes, attachments) in parallel.

    This unified approach simplifies the graph while maintaining
    the same latency characteristics (LLM synthesis dominates).
    """
    config = get_config()
    start_time = time.time()

    logger.info("[Fetch] Fetching data, docs, and account context in parallel...")

    # Prepare results containers
    data_result = {}
    docs_result = {}
    account_result = {}
    errors = []

    # Check if we should fetch account context (notes, attachments via Account RAG)
    intent = state.get("intent", "general")
    company_id = state.get("resolved_company_id")
    should_fetch_account_context = company_id is not None and intent in ACCOUNT_RAG_INTENTS

    # Get state values for fetch functions
    question = state.get("question", "")
    days = state.get("days", config.default_days)
    router_result = state.get("router_result")
    timeout = config.fetch_timeout_seconds

    # Get owner for role-based filtering (from router result)
    owner = getattr(router_result, "owner", None) if router_result else None

    # Execute in parallel using ThreadPoolExecutor
    max_workers = 3 if should_fetch_account_context else 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        futures["data"] = executor.submit(
            _fetch_crm_data,
            question=question,
            intent=intent,
            resolved_company_id=company_id,
            days=days,
            router_result=router_result,
            owner=owner,
        )
        futures["docs"] = executor.submit(_fetch_docs, question=question)

        if should_fetch_account_context:
            futures["account"] = executor.submit(
                _fetch_account_context,
                question=question,
                company_id=company_id,
            )

        # Collect results with timeout
        for name, future in futures.items():
            try:
                result = future.result(timeout=timeout)
                if name == "data":
                    data_result = result
                elif name == "docs":
                    docs_result = result
                elif name == "account":
                    account_result = result
            except FuturesTimeoutError:
                logger.error(f"[Fetch] {name} timed out after {timeout}s")
                errors.append(f"{name}: timeout after {timeout}s")
            except Exception as e:
                logger.error(f"[Fetch] {name} failed: {e}")
                errors.append(f"{name}: {str(e)}")

    # Merge results
    all_sources = (
        data_result.get("sources", [])
        + docs_result.get("docs_sources", [])
        + account_result.get("account_context_sources", [])
    )

    latency_ms = int((time.time() - start_time) * 1000)

    steps = [
        {
            "id": "data",
            "label": "Fetching CRM data",
            "status": "done" if "error" not in data_result else "error",
        },
        {
            "id": "docs",
            "label": "Checking documentation",
            "status": "done" if "error" not in docs_result else "error",
        },
    ]

    if should_fetch_account_context:
        steps.append(
            {
                "id": "account_context",
                "label": "Searching account notes",
                "status": "done" if "error" not in account_result else "error",
            }
        )

    merged = {
        # Data results
        "company_data": data_result.get("company_data"),
        "activities_data": data_result.get("activities_data"),
        "history_data": data_result.get("history_data"),
        "pipeline_data": data_result.get("pipeline_data"),
        "renewals_data": data_result.get("renewals_data"),
        "contacts_data": data_result.get("contacts_data"),
        "groups_data": data_result.get("groups_data"),
        "attachments_data": data_result.get("attachments_data"),
        "resolved_company_id": data_result.get("resolved_company_id"),
        "raw_data": data_result.get("raw_data", {}),
        # Docs results
        "docs_answer": docs_result.get("docs_answer", ""),
        "docs_sources": docs_result.get("docs_sources", []),
        # Account RAG results
        "account_context_answer": account_result.get("account_context_answer", ""),
        "account_context_sources": account_result.get("account_context_sources", []),
        # Combined sources
        "sources": all_sources,
        # Latency
        "fetch_latency_ms": latency_ms,
        # Steps
        "steps": steps,
    }

    if errors:
        merged["error"] = "; ".join(errors)

    logger.info(
        f"[Fetch] Parallel fetch complete in {latency_ms}ms: {len(all_sources)} sources "
        f"(account_context={'yes' if should_fetch_account_context else 'no'})"
    )
    return merged
