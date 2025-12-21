# CRM Synthetic Data Bundle

Drop this folder into your repo as `data/crm/`.

## Files
- `companies.csv`, `contacts.csv`, `activities.csv`, `history.csv`, `opportunities.csv`, `group_members.csv`: from your current dataset.
- `groups.csv`: group definitions for the group_ids referenced in `group_members.csv`.
- `opportunity_descriptions.csv`: **private text** notes per opportunity (for RAG ingestion).
- `attachments.csv`: **private text** attachment metadata + summaries (for RAG ingestion).
- `private_texts.jsonl`: convenience file combining history + opp notes + attachment summaries.
