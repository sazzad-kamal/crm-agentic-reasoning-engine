# Acme CRM Suite – Product Overview

_Last updated: 2025-11-26_

## 1. What is Acme CRM Suite?

Acme CRM Suite is a customer relationship management (CRM) system designed
for small and mid‑sized businesses that need one place to manage:

- Contacts and Companies
- History, Notes, Activities, and Tasks
- Opportunities and Sales Pipeline
- Email Marketing and Follow‑up
- Dashboards and Reports

You can use Acme CRM Suite in the browser; some deployments also sync to
desktop clients and mobile apps so your team can work the way they prefer.

Typical teams using Acme:

- sales and account managers,
- owners and general managers,
- customer success and support,
- marketing for customer communications.

## 2. Core Records and Relationships

At a high level:

- **Company** – a business or organization you sell to or support.
- **Contact** – a person at a company.
- **History / Activity** – what has happened with that company or contact
  (calls, meetings, emails, notes).
- **Opportunity** – a potential sale or renewal.
- **Task** – something you plan to do at a future date.

Common flows:

- A new lead comes in → becomes a **Contact** (and **Company** if needed).
- You call or email them → interactions become **History / Activities**.
- A deal emerges → you create an **Opportunity** tied to that company.
- You schedule follow‑up calls and demos as **Tasks**.
- Over time, your **dashboards** show which deals are moving and which
  accounts are at risk.

## 3. Typical SMB Use Cases

### 3.1 “Everything in one place”

Many small businesses start with spreadsheets and email folders. Acme helps
them move to:

- a single, searchable list of companies/contacts,
- a complete “timeline” of what has happened with each customer,
- a shared view for the whole team instead of siloed inboxes.

### 3.2 “Know what to do today”

Tasks and Activities allow you to see:

- today’s follow‑ups,
- overdue calls,
- upcoming renewals,
- which accounts have gone quiet.

### 3.3 “See the pipeline and renewals”

Opportunities and renewal dates give owners a clear view of:

- expected revenue this month/quarter,
- which deals are stuck,
- which customers are at risk of churning.

## 4. Navigation Overview

Common navigation elements:

- **Companies / Contacts** – the main lists for your customers.
- **Opportunities / Pipeline** – open deals by stage.
- **History / Activities** – calls, meetings, emails, and notes.
- **Tasks** – your to‑do list.
- **Marketing / Campaigns** – customer email campaigns and sequences.
- **Dashboards / Reports** – standard and custom analytics.

Most screens share a few common controls:

- Filters (e.g., owner, status, segment, date range),
- Search boxes,
- Sorting (e.g., next renewal first).

## 5. Data Model (Simplified)

Typical fields:

- Company:
  - Name, Industry, Region
  - Plan / Status (Prospect, Active, Former)
  - Account Owner
  - Renewal Date
- Contact:
  - Name, Email, Phone
  - Role (Decision Maker, Billing, User)
  - Lifecycle Stage (Lead, Customer, Former)
- Opportunity:
  - Name, Company, Owner
  - Stage (e.g., New, Qualified, Proposal, Won/Lost)
  - Value, Expected Close Date
- Activity:
  - Type (Call, Meeting, Email, Note)
  - Date/Time, Owner
  - Linked Company/Contact/Opportunity
- Task:
  - Subject, Due Date, Priority
  - Status (Open, Completed)
  - Linked record

For more detail, see:

- `contacts_companies_and_groups.md`
- `history_activities_and_calendar.md`
- `opportunities_pipeline_and_forecasts.md`
- `email_marketing_campaigns.md`
- `reports_dashboards_and_analytics.md`

## 6. Common Questions

- **“Where should I start if I’m new to Acme CRM Suite?”**  
  Begin with the Companies and Contacts lists to make sure your customer
  data is accurate, then review your pipeline and dashboards.

- **“Do I have to use every module?”**  
  No. Many SMBs start with just Companies, Contacts, and Opportunities, and
  add email or advanced reporting later.

- **“How does Acme CRM Suite relate to my email and calendar?”**  
  Depending on your plan, Acme can log emails and meetings as activities
  and show tasks on your calendar. See `history_activities_and_calendar.md`
  for details.
