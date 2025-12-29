# Opportunities, Pipeline, and Forecasts

_Last updated: 2025-11-26_

## 1. What is an Opportunity?

An **Opportunity** represents a potential sale, expansion, or renewal. It
usually answers:

- What might we sell?
- To which company?
- For how much?
- By when?

Common fields:

- `Name` (e.g., “Acme – Annual Subscription Upgrade”)
- `Company`
- `Owner`
- `Stage`
- `Value` (amount)
- `Expected Close Date`
- `Source` (Referral, Inbound, Outbound, etc.)

## 2. Pipeline Stages

Your pipeline may be configured slightly differently, but a common setup is:

1. New / Discovery
2. Qualified
3. Proposal / Quote
4. Negotiation / Review
5. Closed Won
6. Closed Lost

Each stage should represent a **meaningful step** in your sales process.

Best practices:

- Keep stages simple and easy for the team to understand.
- Avoid too many stages; 5–7 is usually enough.
- Make sure reps know when to move deals to Closed Won or Closed Lost.

## 3. Using the Pipeline Board

The Pipeline board shows opportunities as cards arranged by stage. You can
drag and drop cards to move them between stages.

Common options:

- Filter by `Owner`, `Segment`, `Plan`, or `Region`.
- Sort by `Value` or `Expected Close Date`.
- Show or hide older/closed deals.

Tips:

- Encourage weekly pipeline reviews.
- Use filters so the board remains responsive (e.g., don’t display every
  deal from all time).

## 4. Renewals and Existing Customers

Many SMBs use opportunities for:

- new business (new logo deals),
- expansions (upgrades),
- renewals (contract or subscription renewals).

Typical patterns:

- One **renewal opportunity** per renewal cycle per key account.
- Link renewal opportunities to the correct company and plan.
- Use `Type` (e.g., New Business, Expansion, Renewal) to distinguish deals.

## 5. Forecast Reports

Forecasting helps you answer:

- How much revenue do we expect this month/quarter?
- Where are we ahead or behind?

Forecasts usually combine:

- all **open** opportunities,
- their **stage** (which has a probability),
- their `Expected Close Date`.

For example, your system may treat:

- “Qualified” as 20%,
- “Proposal” as 40%,
- “Negotiation” as 70%.

The forecast multiplies `Value * Probability` for each open deal.

## 6. Common Issues and How to Avoid Them

- **Stale deals (StaleDeals)**  
  Deals sitting in the same stage for months create unrealistic forecasts.  
  Solution: regularly close or re‑qualify older deals.

- **Missing values (DataQualityIssues)**  
  Opportunities without `Value` or `Expected Close Date` skew reports.  
  Solution: make these fields required, or use views to find and fix them.

- **Multiple overlapping renewal deals**  
  Too many renewal opportunities for the same company confuse forecasting.  
  Solution: use a single, clearly labeled renewal opportunity per cycle.

## 7. Getting the Most from Opportunities

For owners and managers:

- Use dashboards to see:
  - Pipeline by stage,
  - Pipeline by owner,
  - Forecast vs target.

For individual reps:

- Focus on:
  - advancing deals to the next stage,
  - scheduling next actions (tasks),
  - keeping close dates realistic.

For more detail, see:
`reports_dashboards_and_analytics.md`.

## 8. Reopening Closed Opportunities

**Can I reopen a closed-lost opportunity?**

Yes. If circumstances change (e.g., the customer comes back, budget is
approved, or a new decision maker takes over), a closed opportunity can
be reopened by changing its stage back to an active stage and updating
the expected close date.

**Best practices for reopening:**

- Only reopen if there's genuine renewed interest from the customer.
- Update the opportunity value if pricing has changed.
- Consider creating a new opportunity instead if significant time has
  passed (6+ months) to keep historical reporting accurate.

---

## 9. Common Questions

- **"When should I close an opportunity as Lost?"**
  Close it when the customer clearly chooses another option, stops
  responding, or postpones the project beyond your forecast horizon.

- **"Should I create separate opportunities for expansions?"**
  Yes, use separate Expansion opportunities so you can track new business,
  renewals, and upgrades separately.

- **"What's the best way to review pipeline as a team?"**
  Use the pipeline board filtered by owner or segment, and review each
  stage weekly, focusing on stuck or stale deals.

- **"Can I reopen a closed-lost opportunity?"**
  Yes, closed opportunities can be reopened by changing the stage back to
  an active stage. See Section 8 for details and best practices.
