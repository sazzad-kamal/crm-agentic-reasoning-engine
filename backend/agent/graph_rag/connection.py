"""Neo4j connection manager with CSV data loading.

Mirrors the DuckDB connection pattern in fetch/sql/connection.py:
thread-local singleton with lazy initialization and CSV loading.
"""

import csv
import logging
import os
import threading
from pathlib import Path

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "csv"

_thread_local = threading.local()


def _load_csv_data(driver) -> None:
    """Load CRM CSV data into Neo4j as nodes and relationships."""
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")

        # Load companies
        csv_file = _CSV_PATH / "companies.csv"
        if csv_file.exists():
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    session.run(
                        "CREATE (c:Company {company_id: $cid, name: $name, domain: $domain, "
                        "status: $status, plan: $plan, account_owner: $owner, industry: $industry, "
                        "segment: $segment, region: $region, renewal_date: $renewal, "
                        "health_status: $health, notes: $notes})",
                        cid=row["company_id"], name=row["name"], domain=row.get("domain", ""),
                        status=row.get("status", ""), plan=row.get("plan", ""),
                        owner=row.get("account_owner", ""), industry=row.get("industry", ""),
                        segment=row.get("segment", ""), region=row.get("region", ""),
                        renewal=row.get("renewal_date", ""), health=row.get("health_status", ""),
                        notes=row.get("notes", ""),
                    )
            logger.debug("Loaded Company nodes from CSV")

        # Load contacts + HAS_CONTACT relationships
        csv_file = _CSV_PATH / "contacts.csv"
        if csv_file.exists():
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    session.run(
                        "CREATE (ct:Contact {contact_id: $cid, first_name: $fn, last_name: $ln, "
                        "email: $email, phone: $phone, job_title: $title, role: $role, "
                        "lifecycle_stage: $stage, notes: $notes})",
                        cid=row["contact_id"], fn=row.get("first_name", ""),
                        ln=row.get("last_name", ""), email=row.get("email", ""),
                        phone=row.get("phone", ""), title=row.get("job_title", ""),
                        role=row.get("role", ""), stage=row.get("lifecycle_stage", ""),
                        notes=row.get("notes", ""),
                    )
                    session.run(
                        "MATCH (c:Company {company_id: $comp_id}), (ct:Contact {contact_id: $cont_id}) "
                        "CREATE (c)-[:HAS_CONTACT]->(ct)",
                        comp_id=row["company_id"], cont_id=row["contact_id"],
                    )
            logger.debug("Loaded Contact nodes and HAS_CONTACT relationships")

        # Load opportunities + HAS_OPPORTUNITY + OWNS_OPPORTUNITY
        csv_file = _CSV_PATH / "opportunities.csv"
        if csv_file.exists():
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    session.run(
                        "CREATE (o:Opportunity {opportunity_id: $oid, name: $name, stage: $stage, "
                        "type: $type, amount: $amount, owner: $owner, "
                        "expected_close_date: $close, notes: $notes})",
                        oid=row["opportunity_id"], name=row.get("name", ""),
                        stage=row.get("stage", ""), type=row.get("type", ""),
                        amount=row.get("amount", "0"), owner=row.get("owner", ""),
                        close=row.get("expected_close_date", ""), notes=row.get("notes", ""),
                    )
                    session.run(
                        "MATCH (c:Company {company_id: $comp_id}), (o:Opportunity {opportunity_id: $oid}) "
                        "CREATE (c)-[:HAS_OPPORTUNITY]->(o)",
                        comp_id=row["company_id"], oid=row["opportunity_id"],
                    )
                    if row.get("contact_id"):
                        session.run(
                            "MATCH (ct:Contact {contact_id: $cont_id}), (o:Opportunity {opportunity_id: $oid}) "
                            "CREATE (ct)-[:OWNS_OPPORTUNITY]->(o)",
                            cont_id=row["contact_id"], oid=row["opportunity_id"],
                        )
            logger.debug("Loaded Opportunity nodes and relationships")

        # Load activities + HAS_ACTIVITY + PARTICIPATED_IN
        csv_file = _CSV_PATH / "activities.csv"
        if csv_file.exists():
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    session.run(
                        "CREATE (a:Activity {activity_id: $aid, type: $type, subject: $subject, "
                        "notes: $notes, due_date: $due, owner: $owner, priority: $priority, "
                        "status: $status})",
                        aid=row["activity_id"], type=row.get("type", ""),
                        subject=row.get("subject", ""), notes=row.get("notes", ""),
                        due=row.get("due_date", ""), owner=row.get("owner", ""),
                        priority=row.get("priority", ""), status=row.get("status", ""),
                    )
                    if row.get("company_id"):
                        session.run(
                            "MATCH (c:Company {company_id: $comp_id}), (a:Activity {activity_id: $aid}) "
                            "CREATE (c)-[:HAS_ACTIVITY]->(a)",
                            comp_id=row["company_id"], aid=row["activity_id"],
                        )
                    if row.get("contact_id"):
                        session.run(
                            "MATCH (ct:Contact {contact_id: $cont_id}), (a:Activity {activity_id: $aid}) "
                            "CREATE (ct)-[:PARTICIPATED_IN]->(a)",
                            cont_id=row["contact_id"], aid=row["activity_id"],
                        )
            logger.debug("Loaded Activity nodes and relationships")

    logger.info("[Neo4j] CRM knowledge graph loaded from CSV files")


def get_driver():
    """Get a thread-local Neo4j driver with CRM data loaded."""
    if not hasattr(_thread_local, "driver") or _thread_local.driver is None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "testpassword")
        _thread_local.driver = GraphDatabase.driver(uri, auth=(user, password))
        _load_csv_data(_thread_local.driver)
        logger.debug("Created new Neo4j driver with CRM knowledge graph")
    return _thread_local.driver


def close_driver() -> None:
    """Close the Neo4j driver for clean shutdown."""
    if hasattr(_thread_local, "driver") and _thread_local.driver is not None:
        _thread_local.driver.close()
        _thread_local.driver = None


__all__ = ["get_driver", "close_driver"]
