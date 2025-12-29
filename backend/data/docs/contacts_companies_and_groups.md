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
- tags for segmenting (e.g., "VIP", "Board Member").

**Can a contact belong to multiple companies?**

Yes. A contact has one **primary company** but can have **secondary
company associations** with relationship types like Consultant, Board
Member, or Advisor. This is useful for consultants, board members, or
contacts who work across multiple client organizations.

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

You can import contacts and companies from spreadsheets, exports from
other CRMs, or signup forms. Imports use CSV files with column mapping
to Acme fields. Duplicate handling options include skip, update existing,
or create new records.

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

## 7. Deleting Records

### Deleting a Contact

When you delete a contact:

- The contact is moved to the Recycle Bin (recoverable for 30 days).
- **Opportunities** linked to that contact are NOT deleted, but the
  contact reference is removed. The opportunity remains linked to its
  company.
- **Activities** logged against the contact are preserved but show
  "[Deleted Contact]" as the participant.
- **Email campaign history** is retained for reporting purposes.

To permanently delete (after 30 days or manual purge):

- All contact data is removed.
- Activity records show "[Deleted]" for the contact name.

### Deleting a Company

When you delete a company:

- All contacts with that company as their **primary** company are also
  deleted (moved to Recycle Bin).
- All opportunities linked to that company are deleted.
- Activities are preserved but show "[Deleted Company]".

**Warning:** Deleting a company is a significant action. Consider changing
the status to "Former" instead if you want to preserve history.

### Handling Duplicates

Duplicate companies can be merged. When merging, one record is kept as
primary and the system combines contacts, opportunities, and activities
from both records into the primary.

---

## 8. Common Questions

- **"Should I create a company for every contact?"**
  For business customers, yes. For one‑off consumers, you can choose to use
  a generic company (e.g., "Individual Customers") if that matches how you
  report.

- **"What's the best way to track multiple brands or locations?"**
  Use separate company records when you need separate ownership, renewals,
  or reporting. Use custom fields or groups to link related companies.

- **"When should I use a Group instead of a report?"**
  Use Groups for lists you'll use often in calls or campaigns (e.g.,
  "Renewals next 60 days"). Use reports when you mainly care about counts
  and totals.

- **"What happens to opportunities when a contact is deleted?"**
  Opportunities are NOT deleted. The contact reference is removed, but the
  opportunity stays linked to its company. See Section 7 for details.
