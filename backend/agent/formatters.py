"""
Context formatting functions for agent prompts.

These functions format CRM data into text sections
for inclusion in LLM prompts.

NOTE: This module re-exports functions from backend.common.formatters
for backwards compatibility. New code should import directly from there.
"""

# Re-export all formatters from the common module
from backend.common.formatters import (
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_docs_section,
    SectionFormatter,
)
