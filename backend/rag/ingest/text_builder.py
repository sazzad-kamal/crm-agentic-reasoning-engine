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
    python -m backend.rag.ingest.text_builder
"""

import json
import random
from pathlib import Path
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from backend.rag.utils import find_csv_dir


# =============================================================================
# Configuration
# =============================================================================

console = Console()


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
        console.print("  [yellow]Warning:[/yellow] opportunities.csv not found, skipping opportunity descriptions")
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
    console.print(f"Building private texts from [cyan]{csv_dir}[/cyan]...")
    
    all_docs = []
    stats = []  # Track stats for summary table
    
    # ---------------------------------------------------------------------------
    # 1. Process history.csv
    # ---------------------------------------------------------------------------
    history_path = csv_dir / "history.csv"
    if history_path.exists():
        console.print("  Processing [bold]history.csv[/bold]...")
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
        
        stats.append(("history.csv", len(history_df)))
    else:
        console.print("  [yellow]Warning:[/yellow] history.csv not found")
    
    # ---------------------------------------------------------------------------
    # 2. Process opportunity_descriptions.csv (or synthesize)
    # ---------------------------------------------------------------------------
    opp_desc_path = csv_dir / "opportunity_descriptions.csv"
    if opp_desc_path.exists():
        console.print("  Processing [bold]opportunity_descriptions.csv[/bold]...")
        opp_desc_df = pd.read_csv(opp_desc_path)
    else:
        console.print("  Synthesizing opportunity descriptions from [bold]opportunities.csv[/bold]...")
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
        
        stats.append(("opportunity_descriptions", len(opp_desc_df)))
    
    # ---------------------------------------------------------------------------
    # 3. Process attachments.csv
    # ---------------------------------------------------------------------------
    attachments_path = csv_dir / "attachments.csv"
    if attachments_path.exists():
        console.print("  Processing [bold]attachments.csv[/bold]...")
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
        
        stats.append(("attachments.csv", len(attachments_df)))
    else:
        console.print("  [yellow]Warning:[/yellow] attachments.csv not found")
    
    # ---------------------------------------------------------------------------
    # 4. Write JSONL output (stable, deterministic order)
    # ---------------------------------------------------------------------------
    # Sort by id for deterministic output
    all_docs.sort(key=lambda x: x["id"])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # Print summary table
    table = Table(title="Build Summary", show_header=True)
    table.add_column("Source", style="cyan")
    table.add_column("Records", justify="right", style="green")
    for source, count in stats:
        table.add_row(source, str(count))
    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{len(all_docs)}[/bold]")
    
    console.print()
    console.print(table)
    console.print(f"\n[green]✓[/green] Wrote to [cyan]{out_path}[/cyan]")


# =============================================================================
# CLI Entrypoint
# =============================================================================

app = typer.Typer(help="Build private_texts.jsonl from CRM CSV files")


@app.command()
def main(
    csv_dir: Path | None = typer.Option(
        None,
        "--csv-dir", "-c",
        help="Directory containing CSV files (auto-detected if not provided)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output JSONL file path (defaults to csv_dir/private_texts.jsonl)",
    ),
):
    """Build private_texts.jsonl from CRM CSV files."""
    console.print(Panel.fit(
        "[bold blue]Private Text Builder[/bold blue] (MVP2)",
        border_style="blue",
    ))
    
    # Find CSV directory
    if csv_dir is None:
        csv_dir = find_csv_dir()
    console.print(f"Using CSV directory: [cyan]{csv_dir}[/cyan]")
    
    # Output path
    out_path = output if output else csv_dir / "private_texts.jsonl"
    
    # Build JSONL
    build_private_texts_jsonl(csv_dir, out_path)
    
    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    app()
    app()
