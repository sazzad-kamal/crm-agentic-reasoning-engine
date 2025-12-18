"""
Private text builder for account-scoped RAG (MVP2).

This module builds private_texts.jsonl from CRM CSV files for private account-scoped RAG ingestion.

Input CSVs:
- history.csv: Calls, emails, meetings, notes
- opportunity_descriptions.csv: Opportunity context/notes
- attachments.csv: Attachment metadata and summaries

Output:
- private_texts.jsonl: Unified JSONL file for ingestion

Usage:
    python -m project1_rag.private_text_builder
"""

import json
import random
from pathlib import Path
from typing import Optional
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Possible CSV directories in priority order
CSV_DIR_CANDIDATES = [
    Path("data/crm"),
    Path("data/csv"),
    Path("docs/csv"),
]

# Required files for MVP2
REQUIRED_FILES = [
    "companies.csv",
    "history.csv",
]


# =============================================================================
# Directory Locator
# =============================================================================

def find_csv_dir() -> Path:
    """
    Find the CSV data directory.
    
    Checks directories in priority order:
    1. data/crm
    2. data/csv
    3. docs/csv
    
    Returns:
        Path to the first existing directory
        
    Raises:
        FileNotFoundError: If no valid directory found
    """
    for candidate in CSV_DIR_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            # Check if required files exist
            has_required = all((candidate / f).exists() for f in REQUIRED_FILES)
            if has_required:
                return candidate
    
    raise FileNotFoundError(
        f"Could not find CSV data directory. "
        f"Please place your CSV files in one of: {[str(p) for p in CSV_DIR_CANDIDATES]}\n"
        f"Required files: {REQUIRED_FILES}"
    )


# =============================================================================
# Opportunity Description Synthesizer
# =============================================================================

def synthesize_opportunity_descriptions(csv_dir: Path) -> pd.DataFrame:
    """
    Synthesize opportunity_descriptions.csv from opportunities.csv.
    
    Creates realistic notes based on opportunity data.
    """
    opp_path = csv_dir / "opportunities.csv"
    if not opp_path.exists():
        print("  Warning: opportunities.csv not found, skipping opportunity descriptions")
        return pd.DataFrame()
    
    opps = pd.read_csv(opp_path)
    
    # Use deterministic seed
    random.seed(42)
    
    # Templates for generating notes
    focus_templates = [
        "streamlining contact management and deduplication",
        "automating follow-ups and reminders",
        "consolidating history/notes into a single customer timeline",
        "segmenting contacts for targeted campaigns",
        "cleaning up pipeline stages and forecasting accuracy",
    ]
    
    risk_templates = [
        "duplicate contacts causing confusion",
        "inconsistent activity logging across the team",
        "low engagement from key contacts",
    ]
    
    next_step_templates = [
        "schedule a 30‑minute admin setup call",
        "import an updated contact list and resolve duplicates",
        "build a renewal dashboard and set weekly review cadence",
        "create a saved group for renewals and start outreach",
    ]
    
    records = []
    for _, row in opps.iterrows():
        opp_id = row.get("opportunity_id", "")
        company_id = row.get("company_id", "")
        contact_id = row.get("primary_contact_id", "")
        company_name = row.get("company_name", company_id)
        stage = row.get("stage", "Unknown")
        value = row.get("value", 0)
        currency = row.get("currency", "USD")
        expected_close = row.get("expected_close_date", "TBD")
        
        # Deterministic selection based on opp_id hash
        hash_val = hash(opp_id)
        focus = focus_templates[hash_val % len(focus_templates)]
        risk = risk_templates[hash_val % len(risk_templates)]
        next_step = next_step_templates[hash_val % len(next_step_templates)]
        
        text = f"""{company_name} opportunity context:
- Stage: {stage}
- Value: {value} {currency}
- Expected close: {expected_close}

Summary: Customer is exploring changes focused on {focus}. Primary risk noted is {risk}. Recommended next step: {next_step}."""
        
        records.append({
            "opportunity_id": opp_id,
            "company_id": company_id,
            "primary_contact_id": contact_id,
            "title": f"Opportunity Notes – {opp_id}",
            "text": text,
            "created_at": row.get("created_at", ""),
            "updated_at": row.get("updated_at", ""),
        })
    
    return pd.DataFrame(records)


# =============================================================================
# Private Text Builder
# =============================================================================

def build_private_texts_jsonl(csv_dir: Path, out_path: Path) -> None:
    """
    Build private_texts.jsonl from CRM CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
        out_path: Output path for JSONL file
        
    This file is used for private account-scoped RAG ingestion.
    """
    print(f"Building private texts from {csv_dir}...")
    
    all_docs = []
    
    # ---------------------------------------------------------------------------
    # 1. Process history.csv
    # ---------------------------------------------------------------------------
    history_path = csv_dir / "history.csv"
    if history_path.exists():
        print("  Processing history.csv...")
        history_df = pd.read_csv(history_path)
        
        for _, row in history_df.iterrows():
            history_id = row.get("history_id", "")
            doc = {
                "id": f"history::{history_id}",
                "company_id": str(row.get("company_id", "")),
                "contact_id": str(row.get("contact_id", "")),
                "opportunity_id": row.get("opportunity_id") if pd.notna(row.get("opportunity_id")) else None,
                "type": "history",
                "title": str(row.get("subject", "")),
                "text": str(row.get("description", "")),
                "metadata": {
                    "history_type": str(row.get("type", "")),
                    "occurred_at": str(row.get("occurred_at", "")),
                    "owner": str(row.get("owner", "")),
                    "source": str(row.get("source", "")),
                },
            }
            all_docs.append(doc)
        
        print(f"    -> {len(history_df)} history records")
    else:
        print("  Warning: history.csv not found")
    
    # ---------------------------------------------------------------------------
    # 2. Process opportunity_descriptions.csv (or synthesize)
    # ---------------------------------------------------------------------------
    opp_desc_path = csv_dir / "opportunity_descriptions.csv"
    if opp_desc_path.exists():
        print("  Processing opportunity_descriptions.csv...")
        opp_desc_df = pd.read_csv(opp_desc_path)
    else:
        print("  Synthesizing opportunity descriptions from opportunities.csv...")
        opp_desc_df = synthesize_opportunity_descriptions(csv_dir)
    
    if not opp_desc_df.empty:
        for _, row in opp_desc_df.iterrows():
            opp_id = row.get("opportunity_id", "")
            doc = {
                "id": f"opp_note::{opp_id}",
                "company_id": str(row.get("company_id", "")),
                "contact_id": str(row.get("primary_contact_id", "")),
                "opportunity_id": str(opp_id),
                "type": "opportunity_note",
                "title": str(row.get("title", f"Opportunity Notes – {opp_id}")),
                "text": str(row.get("text", "")),
                "metadata": {
                    "updated_at": str(row.get("updated_at", "")),
                },
            }
            all_docs.append(doc)
        
        print(f"    -> {len(opp_desc_df)} opportunity notes")
    
    # ---------------------------------------------------------------------------
    # 3. Process attachments.csv
    # ---------------------------------------------------------------------------
    attachments_path = csv_dir / "attachments.csv"
    if attachments_path.exists():
        print("  Processing attachments.csv...")
        attachments_df = pd.read_csv(attachments_path)
        
        for _, row in attachments_df.iterrows():
            att_id = row.get("attachment_id", "")
            doc = {
                "id": f"attachment::{att_id}",
                "company_id": str(row.get("company_id", "")),
                "contact_id": str(row.get("contact_id", "")),
                "opportunity_id": row.get("opportunity_id") if pd.notna(row.get("opportunity_id")) else None,
                "type": "attachment",
                "title": str(row.get("title", "")),
                "text": str(row.get("summary", "")),
                "metadata": {
                    "file_type": str(row.get("file_type", "")),
                    "created_at": str(row.get("created_at", "")),
                },
            }
            all_docs.append(doc)
        
        print(f"    -> {len(attachments_df)} attachments")
    else:
        print("  Warning: attachments.csv not found")
    
    # ---------------------------------------------------------------------------
    # 4. Write JSONL output (stable, deterministic order)
    # ---------------------------------------------------------------------------
    # Sort by id for deterministic output
    all_docs.sort(key=lambda x: x["id"])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    print(f"\nWrote {len(all_docs)} documents to {out_path}")


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    """Main entrypoint for private text building."""
    print("=" * 60)
    print("Private Text Builder (MVP2)")
    print("=" * 60)
    
    # Find CSV directory
    csv_dir = find_csv_dir()
    print(f"Using CSV directory: {csv_dir}")
    
    # Output path
    out_path = csv_dir / "private_texts.jsonl"
    
    # Build JSONL
    build_private_texts_jsonl(csv_dir, out_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
