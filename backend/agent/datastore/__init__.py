"""
CRM Data Store package.

Provides DuckDB-based CRM data access with lazy loading.
"""

from backend.agent.datastore.base import (
    CRMDataStoreBase,
    get_csv_base_path,
    _get_datastore_instance,
    CSV_TABLES,
    REQUIRED_TABLES,
)
from backend.agent.datastore.companies import CompanyMixin
from backend.agent.datastore.contacts import ContactMixin
from backend.agent.datastore.pipeline import PipelineMixin
from backend.agent.datastore.activities import ActivityMixin
from backend.agent.datastore.analytics import AnalyticsMixin


class CRMDataStore(
    CompanyMixin,
    ContactMixin,
    PipelineMixin,
    ActivityMixin,
    AnalyticsMixin,
    CRMDataStoreBase,
):
    """
    DuckDB-based CRM data store with lazy loading.

    Combines all domain-specific mixins for complete CRM functionality.

    Supports context manager for automatic cleanup:
        with CRMDataStore() as store:
            data = store.get_company("ACME-MFG")
    """

    pass


def get_datastore() -> CRMDataStore:
    """Get a thread-local CRMDataStore instance.

    Each thread gets its own DuckDB connection to avoid
    concurrent access issues.
    """
    return _get_datastore_instance(CRMDataStore)


__all__ = ["CRMDataStore", "get_datastore", "get_csv_base_path", "CSV_TABLES", "REQUIRED_TABLES"]
