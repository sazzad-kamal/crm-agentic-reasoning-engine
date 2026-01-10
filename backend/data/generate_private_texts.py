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
    python generate_private_texts.py [--merge-opportunities]

Options:
    --merge-opportunities  Merge opportunity_descriptions.csv into opportunities.csv first
"""

import argparse
import csv
import json
from pathlib import Path

CSV_DIR = Path(__file__).parent / "csv"


def merge_opportunity_descriptions() -> None:
    """Merge opportunity_descriptions.csv text into opportunities.csv as notes column."""
    opps_path = CSV_DIR / "opportunities.csv"
    descs_path = CSV_DIR / "opportunity_descriptions.csv"

    if not descs_path.exists():
        print("No opportunity_descriptions.csv found, skipping merge")
        return

    # Read opportunity descriptions
    desc_map: dict[str, str] = {}
    with open(descs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opp_id = row.get("opportunity_id", "")
            text = row.get("text", "")
            if opp_id and text:
                desc_map[opp_id] = text

    print(f"Loaded {len(desc_map)} opportunity descriptions")

    # Read opportunities and add notes column
    rows = []
    fieldnames = []
    with open(opps_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "notes" not in fieldnames:
            fieldnames.append("notes")
        for row in reader:
            opp_id = row.get("opportunity_id", "")
            row["notes"] = desc_map.get(opp_id, "")
            rows.append(row)

    # Write updated opportunities.csv
    with open(opps_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated opportunities.csv with notes column ({len(rows)} rows)")


def generate_private_texts() -> int:
    """Generate private_texts.jsonl from source CSVs."""
    output_path = CSV_DIR / "private_texts.jsonl"
    records = []

    # 1. Companies (description field)
    companies_path = CSV_DIR / "companies.csv"
    if companies_path.exists():
        with open(companies_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                desc = row.get("description", "").strip()
                if desc:
                    records.append({
                        "id": f"company::{row.get('company_id', '')}",
                        "company_id": row.get("company_id", ""),
                        "contact_id": "",
                        "opportunity_id": "",
                        "type": "company",
                        "title": row.get("name", ""),
                        "text": desc,
                        "metadata": {
                            "status": row.get("status", ""),
                            "plan": row.get("plan", ""),
                            "industry": row.get("industry", ""),
                        },
                    })
        print(f"  Companies: {sum(1 for r in records if r['type'] == 'company')} records")

    # 2. Contacts (notes field)
    contacts_path = CSV_DIR / "contacts.csv"
    if contacts_path.exists():
        count_before = len(records)
        with open(contacts_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                notes = row.get("notes", "").strip()
                if notes:
                    records.append({
                        "id": f"contact::{row.get('contact_id', '')}",
                        "company_id": row.get("company_id", ""),
                        "contact_id": row.get("contact_id", ""),
                        "opportunity_id": "",
                        "type": "contact",
                        "title": f"{row.get('first_name', '')} {row.get('last_name', '')}",
                        "text": notes,
                        "metadata": {
                            "job_title": row.get("job_title", ""),
                            "role": row.get("role", ""),
                            "lifecycle_stage": row.get("lifecycle_stage", ""),
                        },
                    })
        print(f"  Contacts: {len(records) - count_before} records")

    # 3. Opportunities (notes field)
    opps_path = CSV_DIR / "opportunities.csv"
    if opps_path.exists():
        count_before = len(records)
        with open(opps_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                notes = row.get("notes", "").strip()
                if notes:
                    records.append({
                        "id": f"opportunity::{row.get('opportunity_id', '')}",
                        "company_id": row.get("company_id", ""),
                        "contact_id": row.get("primary_contact_id", ""),
                        "opportunity_id": row.get("opportunity_id", ""),
                        "type": "opportunity",
                        "title": row.get("name", ""),
                        "text": notes,
                        "metadata": {
                            "stage": row.get("stage", ""),
                            "value": row.get("value", ""),
                            "created_at": row.get("created_at", ""),
                        },
                    })
        print(f"  Opportunities: {len(records) - count_before} records")

    # 4. History (description field)
    history_path = CSV_DIR / "history.csv"
    if history_path.exists():
        count_before = len(records)
        with open(history_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                desc = row.get("description", "").strip()
                if desc:
                    records.append({
                        "id": f"history::{row.get('history_id', '')}",
                        "company_id": row.get("company_id", ""),
                        "contact_id": row.get("contact_id", ""),
                        "opportunity_id": row.get("opportunity_id", ""),
                        "type": "history",
                        "title": row.get("subject", ""),
                        "text": desc,
                        "metadata": {
                            "history_type": row.get("type", ""),
                            "occurred_at": row.get("occurred_at", ""),
                            "owner": row.get("owner", ""),
                            "source": row.get("source", ""),
                        },
                    })
        print(f"  History: {len(records) - count_before} records")

    # 5. Activities (description field)
    activities_path = CSV_DIR / "activities.csv"
    if activities_path.exists():
        count_before = len(records)
        with open(activities_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                desc = row.get("description", "").strip()
                if desc:
                    records.append({
                        "id": f"activity::{row.get('activity_id', '')}",
                        "company_id": row.get("company_id", ""),
                        "contact_id": row.get("contact_id", ""),
                        "opportunity_id": row.get("opportunity_id", ""),
                        "type": "activity",
                        "title": row.get("subject", ""),
                        "text": desc,
                        "metadata": {
                            "activity_type": row.get("type", ""),
                            "due_datetime": row.get("due_datetime", ""),
                            "owner": row.get("owner", ""),
                            "status": row.get("status", ""),
                            "priority": row.get("priority", ""),
                        },
                    })
        print(f"  Activities: {len(records) - count_before} records")

    # 6. Attachments (summary field)
    attachments_path = CSV_DIR / "attachments.csv"
    if attachments_path.exists():
        count_before = len(records)
        with open(attachments_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary = row.get("summary", "").strip()
                if summary:
                    records.append({
                        "id": f"attachment::{row.get('attachment_id', '')}",
                        "company_id": row.get("company_id", ""),
                        "contact_id": row.get("contact_id", ""),
                        "opportunity_id": row.get("opportunity_id", ""),
                        "type": "attachment",
                        "title": row.get("title", ""),
                        "text": summary,
                        "metadata": {
                            "file_type": row.get("file_type", ""),
                            "created_at": row.get("created_at", ""),
                        },
                    })
        print(f"  Attachments: {len(records) - count_before} records")

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nGenerated {output_path.name} with {len(records)} total records")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Generate private_texts.jsonl from source CSVs")
    parser.add_argument(
        "--merge-opportunities",
        action="store_true",
        help="Merge opportunity_descriptions.csv into opportunities.csv first",
    )
    args = parser.parse_args()

    if args.merge_opportunities:
        print("Step 1: Merging opportunity descriptions...")
        merge_opportunity_descriptions()
        print()

    print("Generating private_texts.jsonl...")
    generate_private_texts()


if __name__ == "__main__":
    main()
