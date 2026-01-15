# CRM Synthetic Data Bundle

Drop this folder into your repo as `data/crm/`.

## Files

### Core Tables
- `companies.csv`: Company records (includes `description` column for RAG)
- `contacts.csv`: Contact records linked to companies (includes `notes` column for RAG)
- `opportunities.csv`: Sales opportunities (includes `notes` column for RAG)
- `activities.csv`: Scheduled tasks and activities (includes `description` column for RAG)
- `history.csv`: Completed interactions (calls, emails, meetings, notes) - includes `description` column for RAG

### Generated File
- `texts.jsonl`: Combined text content for RAG search (generated from companies, contacts, opportunities, history, activities)

## Regenerating texts.jsonl

Run from the backend/data directory:
```bash
python generate_texts.py
```
