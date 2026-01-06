"""Support utilities for the graph execution layer."""

from backend.agent.nodes.support.memory import (
    clear_session,
    format_history_for_prompt,
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
    # Memory
    "clear_session",
    "format_history_for_prompt",
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
