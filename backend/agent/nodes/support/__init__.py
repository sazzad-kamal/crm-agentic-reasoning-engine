"""
Support utilities for the graph execution layer.

Provides caching, memory, streaming, formatting, and audit logging.
"""

from backend.agent.nodes.support.session import (
    get_checkpointer,
    get_session_state,
    get_session_messages,
    build_thread_config,
    make_cache_key,
    get_cached_result,
    set_cached_result,
    clear_query_cache,
)
from backend.agent.nodes.support.memory import (
    clear_session,
    format_history_for_prompt,
)
from backend.agent.nodes.support.streaming import (
    stream_agent,
    StreamEvent,
    format_sse,
    serialize_for_json,
)
from backend.agent.nodes.support.audit import (
    AgentAuditEntry,
    AgentAuditLogger,
    get_audit_logger,
)
from backend.agent.nodes.support.formatters import (
    SectionFormatter,
    FORMATTERS,
    format_section,
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_contacts_section,
    format_groups_section,
    format_attachments_section,
    format_docs_section,
    format_account_context_section,
    format_conversation_history_section,
)

__all__ = [
    # Session
    "get_checkpointer",
    "get_session_state",
    "get_session_messages",
    "build_thread_config",
    "make_cache_key",
    "get_cached_result",
    "set_cached_result",
    "clear_query_cache",
    # Memory
    "clear_session",
    "format_history_for_prompt",
    # Streaming
    "stream_agent",
    "StreamEvent",
    "format_sse",
    "serialize_for_json",
    # Audit
    "AgentAuditEntry",
    "AgentAuditLogger",
    "get_audit_logger",
    # Formatters
    "SectionFormatter",
    "FORMATTERS",
    "format_section",
    "format_company_section",
    "format_activities_section",
    "format_history_section",
    "format_pipeline_section",
    "format_renewals_section",
    "format_contacts_section",
    "format_groups_section",
    "format_attachments_section",
    "format_docs_section",
    "format_account_context_section",
    "format_conversation_history_section",
]
