# Contacts, Companies, and Groups

_Last updated: 2025-11-26_

## 1. Companies

A **Company** record represents an organization.

Common fields:

- `Company Name`
- `Primary Domain` (e.g., `customer.com`)
- `Address` (street, city, region, country)
- `Status` (Prospect, Active, Former)
- `Plan` (Free, Standard, Pro, Enterprise, or custom)
- `Account Owner` (internal user)
- `Renewal Date`
- `Industry`, `Segment`, `Region`

A company is typically the main anchor for:

- Opportunities (deals),
- Activities and Tasks,
- Tickets (if you track support),
- Billing and renewal data.

## 2. Contacts

A **Contact** is a person at a company.

Typical fields:

- `First Name`, `Last Name`
- `Email`, `Phone`
- `Job Title`, `Role` (Decision Maker, Billing, Technical, Champion)
- `Lifecycle` (Lead, Customer, Former Customer)
- `Preferred Language` or `Time Zone` (optional)

Each contact should normally be linked to a **primary company**. You can
also store:

- additional company associations as notes or custom fields,
- tags for segmenting (e.g., “VIP”, “Board Member”).

## 3. Groups and Segmentation

**Groups** (sometimes called lists or segments) let you save commonly used
filters. Examples:

- All active customers in Canada.
- All Pro plan accounts renewing in the next 90 days.
- All contacts tagged as Champion in Manufacturing.

Groups are used for:

- focusing call and task lists,
- building email campaigns,
- driving custom dashboards.

Group definitions are usually:

- a base record type (Company or Contact),
- one or more filters,
- optional sort order.

## 4. Best Practices for SMB Teams

### 4.1 Keep company names clean

Use consistent naming (e.g., “Acme Inc.”, not “Acme”, “ACME Inc”, “Acme
Canada”). This helps with:

- duplicate detection,
- searching,
- reporting.

### 4.2 Use domains and emails to avoid duplicates

Make sure:

- `Primary Domain` is set for companies,
- `Email` is unique per contact whenever possible.

If the CRM warns about duplicates when importing or adding records, review
them carefully.

### 4.3 Assign clear ownership

Every active account should have an `Account Owner`. This determines:

- who sees the account on their “My Accounts” views,
- who is responsible for renewals and upsells,
- who receives certain alerts.

### 4.4 Use groups for repeatable work

Instead of manually filtering every time:

- create a group for “Renewals Next 60 Days”,
- create a group for “New Leads This Week”,
- create a group for “High‑value Accounts with No Activity in 30 Days”.

Then use these groups to drive tasks, calls, and campaigns.

## 5. Importing Contacts and Companies

You can import contacts and companies from:

- spreadsheets,
- exports from other CRMs,
- signup forms (depending on your setup).

Basic steps:

1. Download or prepare a CSV file.
2. Map your columns to Acme fields (Company Name, Email, etc.).
3. Choose how to handle duplicates (skip, update, or create new).
4. Run the import and review the summary.

See `faq_data_import_export.md` for more on imports and exports.

## 6. Linking to Other Modules

- **Opportunities** are always tied to a company and often have primary
  contacts.
- **Activities and Tasks** can be created from the company or contact
  pages.
- **Email Campaigns** often target contact groups (e.g., all Champions at
  Pro plan accounts).

For more detailed workflows:

- see `history_activities_and_calendar.md`,
- and `opportunities_pipeline_and_forecasts.md`.

## 7. Common Questions

- **“Should I create a company for every contact?”**  
  For business customers, yes. For one‑off consumers, you can choose to use
  a generic company (e.g., “Individual Customers”) if that matches how you
  report.

- **“What’s the best way to track multiple brands or locations?”**  
  Use separate company records when you need separate ownership, renewals,
  or reporting. Use custom fields or groups to link related companies.

- **“When should I use a Group instead of a report?”**  
  Use Groups for lists you’ll use often in calls or campaigns (e.g.,
  “Renewals next 60 days”). Use reports when you mainly care about counts
  and totals.
