# System Performance and Limits

_Last updated: 2025-11-26_

This document describes how certain configuration choices and data volumes
can affect the behaviour and performance of Acme CRM Suite.

It is intended as a high‑level guide to help you understand why some
operations may take longer than others, and what general patterns to be
aware of.

---

## 1. Reports and Dashboards

Reports and dashboards are built on top of the underlying data in your
database. The time it takes to run a report depends mainly on:

- the number of records being scanned (date range, filters),
- the number of grouping and aggregation fields,
- and any additional calculations or joins.

In general:

- **Broader date ranges** (for example, “all time” instead of “last 12
  months”) require scanning more rows.
- **Multiple grouping fields** (for example, grouping by owner, region,
  and product at the same time) produce more groups to compute.
- **Complex filters** (several conditions combined) add extra work.

If a particular report feels heavy or slow in a production system
(**SlowReports**), common approaches are:

- limiting the date range (for example, last 12–24 months),
- reducing the number of grouping fields,
- running separate, more targeted reports instead of a single large one.

---

## 2. Activities, History, and Log Volume

Activities and history records (calls, meetings, emails, notes, system
events) typically grow faster than other tables. Over time, this can affect:

- how quickly large activity lists load,
- how long it takes to generate certain activity‑based reports.

In many deployments, this is managed by:

- using date filters when viewing activities (for example, “last 90 days”),
- relying on saved views for common time windows,
- using summary‑style reports instead of listing every historical entry at
  once.

The CRM data model is designed to handle large volumes, but very broad,
unfiltered queries over all history for all time can take longer than
narrow, filtered ones.

---

## 3. Attachments and Storage

Acme CRM Suite supports attachments (documents, images, and other files)
linked to companies, contacts, opportunities, and other entities. Typical
considerations:

- Very large files consume more storage and may take longer to upload or
  download.
- Large numbers of attachments on a single record can increase load times
  for that record’s document view.
- Some deployments enforce maximum file size or total storage quotas.

If you work with many large attachments in a production environment, it is
common to:

- store only the most relevant documents in the CRM,
- archive older or rarely used files externally,
- and apply sensible file‑size limits.

---

## 4. List Views and Pagination

List views (for example, Companies, Contacts, Opportunities, Activities)
are usually paginated. Performance depends on:

- how many columns are displayed,
- how many filters and sorts are applied,
- whether there are additional calculated fields.

To keep list views responsive in larger deployments, it is typical to:

- filter by owner, segment, or status rather than loading all records,
- use saved views for frequently used filters,
- and rely on pagination rather than attempting to show every record on
  one page.

---

## 5. Integrations and API Usage

If you use integrations or automated processes (for example, connecting
the CRM to other systems through APIs), overall performance can be
affected by:

- the volume and frequency of API calls,
- the complexity of each operation (single‑record vs bulk operations),
- and how error handling and retries are implemented.

Many environments apply:

- rate limits to protect overall system stability,
- recommended patterns such as batching updates instead of updating one
  record at a time in quick succession.

For integration or API design, it is generally better to:

- avoid very frequent small updates where possible,
- use incremental changes and batching,
- and monitor for error responses or throttling signals.

---

## 6. General Notes

The CRM is designed to operate reliably with typical small and mid‑sized
business workloads. In practice:

- focused queries with reasonable filters and date ranges tend to perform
  better than unbounded ones,
- lists and reports are easier to interpret when they are scoped to a
  specific owner, segment, or period,
- and background jobs (such as imports, exports, or large maintenance
  tasks) may run for a longer time depending on data volume.

This document is not an exhaustive performance guide, but outlines the
main patterns that can influence how quickly certain operations complete.

---

## 7. Common Questions

- **“Why did my report suddenly feel slower?”**  
  Check whether you recently widened the date range, added extra grouping
  fields, or removed filters that kept the dataset small.

- **“Do I need to archive old data?”**  
  Most SMBs can keep years of history online. If you see performance
  issues, start with filters and saved views before considering data
  archival.

- **“Can I see hard limits for my deployment?”**  
  Your Acme administrator or implementation partner can provide details
  about any plan‑specific limits or quotas.
