#!/usr/bin/env python3
"""
Generate private_texts.jsonl from source CSVs.

Sources:
- companies.csv (description field)
- contacts.csv (notes field)
- opportunities.csv (notes field)
- history.csv (description field)
- activities.csv (description field)
- attachments.csv (summary field)

Usage:
    python generate_private_texts.py [--input-dir DIR]

Options:
    --input-dir DIR  Input directory for CSVs (default: ./csv)
"""

import argparse
import csv
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CSV_DIR = Path(__file__).parent / "csv"


@dataclass
class CsvSource:
    """Configuration for a CSV source file."""

    filename: str
    id_field: str
    text_field: str
    record_type: str
    title_fn: Callable[[dict[str, str]], str]
    metadata_fn: Callable[[dict[str, str]], dict[str, str]]
    # Optional: override which ID fields to extract
    company_id_field: str = "company_id"
    contact_id_field: str = "contact_id"
    opportunity_id_field: str = "opportunity_id"


# Define all CSV sources in one place
SOURCES: list[CsvSource] = [
    CsvSource(
        filename="companies.csv",
        id_field="company_id",
        text_field="description",
        record_type="company",
        title_fn=lambda r: r.get("name", ""),
        metadata_fn=lambda r: {
            "status": r.get("status", ""),
            "plan": r.get("plan", ""),
            "industry": r.get("industry", ""),
        },
        contact_id_field="",  # Companies don't have contact_id
        opportunity_id_field="",  # Companies don't have opportunity_id
    ),
    CsvSource(
        filename="contacts.csv",
        id_field="contact_id",
        text_field="notes",
        record_type="contact",
        title_fn=lambda r: f"{r.get('first_name', '')} {r.get('last_name', '')}".strip(),
        metadata_fn=lambda r: {
            "job_title": r.get("job_title", ""),
            "role": r.get("role", ""),
            "lifecycle_stage": r.get("lifecycle_stage", ""),
        },
        opportunity_id_field="",  # Contacts don't have opportunity_id directly
    ),
    CsvSource(
        filename="opportunities.csv",
        id_field="opportunity_id",
        text_field="notes",
        record_type="opportunity",
        title_fn=lambda r: r.get("name", ""),
        metadata_fn=lambda r: {
            "stage": r.get("stage", ""),
            "value": r.get("value", ""),
            "created_at": r.get("created_at", ""),
        },
        contact_id_field="primary_contact_id",
    ),
    CsvSource(
        filename="history.csv",
        id_field="history_id",
        text_field="description",
        record_type="history",
        title_fn=lambda r: r.get("subject", ""),
        metadata_fn=lambda r: {
            "history_type": r.get("type", ""),
            "occurred_at": r.get("occurred_at", ""),
            "owner": r.get("owner", ""),
            "source": r.get("source", ""),
        },
    ),
    CsvSource(
        filename="activities.csv",
        id_field="activity_id",
        text_field="description",
        record_type="activity",
        title_fn=lambda r: r.get("subject", ""),
        metadata_fn=lambda r: {
            "activity_type": r.get("type", ""),
            "due_datetime": r.get("due_datetime", ""),
            "owner": r.get("owner", ""),
            "status": r.get("status", ""),
            "priority": r.get("priority", ""),
        },
    ),
    CsvSource(
        filename="attachments.csv",
        id_field="attachment_id",
        text_field="summary",
        record_type="attachment",
        title_fn=lambda r: r.get("title", ""),
        metadata_fn=lambda r: {
            "file_type": r.get("file_type", ""),
            "created_at": r.get("created_at", ""),
        },
    ),
]


def process_csv_source(source: CsvSource, csv_dir: Path) -> list[dict]:
    """Process a single CSV source and return records."""
    csv_path = csv_dir / source.filename
    if not csv_path.exists():
        return []

    records = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(source.text_field, "").strip()
            if not text:
                continue

            records.append({
                "id": f"{source.record_type}::{row.get(source.id_field, '')}",
                "company_id": row.get(source.company_id_field, "") if source.company_id_field else "",
                "contact_id": row.get(source.contact_id_field, "") if source.contact_id_field else "",
                "opportunity_id": row.get(source.opportunity_id_field, "") if source.opportunity_id_field else "",
                "type": source.record_type,
                "title": source.title_fn(row),
                "text": text,
                "metadata": source.metadata_fn(row),
            })

    return records


def generate_private_texts(csv_dir: Path) -> int:
    """Generate private_texts.jsonl from source CSVs."""
    output_path = csv_dir / "private_texts.jsonl"
    all_records = []

    print("Processing sources:")
    for source in SOURCES:
        records = process_csv_source(source, csv_dir)
        if records:
            print(f"  {source.record_type.capitalize()}: {len(records)} records")
            all_records.extend(records)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nGenerated {output_path.name} with {len(all_records)} total records")
    return len(all_records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate private_texts.jsonl from source CSVs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=CSV_DIR,
        help=f"Input directory for CSVs (default: {CSV_DIR})",
    )
    args = parser.parse_args()

    csv_dir = args.input_dir
    if not csv_dir.exists():
        print(f"Error: Input directory does not exist: {csv_dir}", file=sys.stderr)
        return 1

    print("Generating private_texts.jsonl...")
    generate_private_texts(csv_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
