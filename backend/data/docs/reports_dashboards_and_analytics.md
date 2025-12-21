# Reports, Dashboards, and Analytics

_Last updated: 2025-11-26_

## 1. Standard Dashboards

Acme CRM Suite includes a set of pre‑built dashboards to help SMBs get
started quickly:

- **Sales Overview**
  - Open pipeline by stage
  - Won deals by month
  - New opportunities created
- **Customer Health**
  - Accounts by plan and region
  - Accounts with upcoming renewals
  - Accounts with low recent activity
- **Activity Overview**
  - Activities by type (calls, meetings, emails)
  - Activities by owner
  - Companies with no activity in a given period
- **Email Performance**
  - Campaign open and click‑through rates
  - Unsubscribe and bounce metrics

Each dashboard tile can usually be filtered by date range and owner.

## 2. Custom Reports

The **Report Builder** allows you to create your own reports by:

- choosing a base record type:
  - Companies
  - Contacts
  - Opportunities
  - Activities
  - Tasks
- applying filters:
  - date ranges,
  - owners,
  - segment or region,
  - plan,
  - status or stage.
- choosing columns and grouping:
  - group by owner, stage, region, etc.
- choosing metrics:
  - count,
  - sum (e.g., total deal value),
  - average (e.g., average number of activities).

Examples:

- “Opportunities created this quarter by owner.”
- “Active accounts with no activity in the last 60 days.”
- “Activities per user per week.”

## 3. Performance Tips for SMB Users

Most small businesses won’t hit hard technical limits, but some practices
help keep reports responsive and avoid **SlowReports**:

- Use **reasonable date ranges** (e.g., last 12 months, this year).
- Avoid grouping by extremely detailed fields (like full email address) when
  you just need totals by owner or region.
- Save commonly used filters and dashboards instead of building from
  scratch each time.

If a report feels slow:

- try removing one or two filters or groupings,
- or narrow the date range.

## 4. Exporting Data

Admins can export data for further analysis in spreadsheets or BI tools.

Typical exports:

- contact lists,
- company lists,
- opportunities,
- activity logs,
- email campaign results.

Exports usually appear as CSV files available for secure download for a
limited time.

For details on imports and exports, see:
`faq_data_import_export.md`.

## 5. Using Analytics in Daily Work

Owners and managers:

- use pipeline and forecast reports for planning,
- review account health and renewal dashboards.

Sales reps:

- use activity reports to make sure they’re staying in touch with
  important accounts,
- focus on opportunities nearing close dates.

Customer success:

- use “accounts with low activity” and “upcoming renewals” to prioritize
  outreach,
- combine email metrics with activity history to decide who needs extra
  help.

## 6. Common Questions

- **“Why doesn’t this dashboard match my CSV export exactly?”**  
  Check:
  - whether the filters and date ranges are the same,
  - whether the dashboard uses “as of” snapshots vs current data.

- **“Can I build any report I want?”**  
  The report builder covers most common SMB needs. For very specialized
  reports, export data and use your preferred external tools.

- **“How do I monitor accounts with low activity?”**  
  Use a dashboard tile or custom report based on activity date; many SMBs
  create a view for “Active accounts with no activity in the last 30 or 60
  days.”
