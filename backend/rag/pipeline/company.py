"""
Company resolution utilities for account-scoped pipelines.

This module handles company data loading and resolution from company names
using the companies CSV data.

NOTE: This module re-exports functions from backend.common.company_resolver
for backwards compatibility. New code should import directly from there.
"""

# Re-export from common module
from backend.common.company_resolver import (
    load_companies_df,
    resolve_company_id,
    get_company_name,
    get_company_matches,
    CompanyResolver,
    get_resolver,
    clear_cache,
)
