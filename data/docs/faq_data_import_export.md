# Importing and Exporting Data (FAQ)

_Last updated: 2025-11-26_

This FAQ covers common questions about bringing data into Acme CRM Suite
and exporting it for backup or analysis.

## 1. What can I import?

Most deployments allow you to import:

- Companies
- Contacts
- Opportunities
- Activities or history (in some cases)

Imports are usually done from CSV files prepared in a spreadsheet tool.

## 2. Basic import steps

1. Download or prepare a CSV file with clear column headers.
2. Choose the target record type (Companies, Contacts, etc.).
3. Map your columns to Acme fields (e.g., Company Name, Email, Owner).
4. Decide how to handle duplicates (skip, update existing, or create new).
5. Run the import and review the summary report.

If you see many skipped rows, check the error messages for hints
(e.g., missing required fields or invalid formats).

## 3. How are duplicates handled?

Depending on configuration, the import wizard can:

- match companies by name and/or primary domain,
- match contacts by email address.

Common options:

- **Skip duplicates** – keep your existing records.
- **Update duplicates** – update matched records with values from the file.
- **Always create new** – rarely recommended for ongoing imports.

## 4. What can I export?

Admins can usually export:

- company lists,
- contact lists,
- opportunities and pipeline data,
- activity logs,
- campaign results.

Exports are provided as CSV files that you can open in Excel or
another analysis tool.

## 5. How do exports affect performance or security?

Exports may:

- be limited to certain roles (e.g., admins),
- be time‑limited download links,
- be logged for audit purposes.

If your environment handles sensitive data, follow your company’s policies
on where exported files are stored and who can access them.

## 6. When should I contact my administrator?

Contact your admin or support team if:

- you are importing more than a few thousand rows at once,
- you are migrating from another CRM,
- you are unsure how duplicates will be handled,
- you need a custom or scheduled export.

For more detail on analytics and reporting, see:
`reports_dashboards_and_analytics.md`.
