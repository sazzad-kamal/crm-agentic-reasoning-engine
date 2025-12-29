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

---

## 10. Identifying At-Risk Deals

Deals can become "at risk" for several reasons. The system automatically
flags deals based on these indicators:

### Stale Deals (Days in Stage)

Deals that sit in the same stage too long are flagged:

- **45+ days in Discovery** – Qualification is stalled
- **45+ days in Proposal** – Waiting too long for customer response
- **60+ days in Negotiation** – Contract review is taking too long

**What to do with stale deals:**

1. Review the `next_step` field – is there a clear action?
2. Check recent activities – when was the last customer contact?
3. Decide: advance, re-qualify, or close as lost

### Health Indicators

Beyond stage duration, consider:

- **No activities in 30+ days** – Engagement has dropped
- **Missed expected close date** – Deal is overdue
- **No next step defined** – No clear path forward

**Pro tip:** Ask "Why is this deal stalled?" to get a summary of
blockers from notes and history.

---

## 11. Weighted Pipeline Forecast

The system calculates a **weighted forecast** using stage probabilities:

| Stage | Probability |
|-------|-------------|
| New / Discovery | 10% |
| Qualified | 25% |
| Proposal | 50% |
| Negotiation | 75% |
| Closed Won | 100% |
| Closed Lost | 0% |

**Weighted Value** = Deal Value × Stage Probability

**Example:**
A $32,000 deal in Negotiation contributes $24,000 to weighted forecast
(32,000 × 0.75).

### Understanding Your Forecast

- **Total Pipeline** – Sum of all open deal values
- **Weighted Pipeline** – Risk-adjusted forecast (more realistic)
- **Best Case** – All open deals close at full value
- **Committed** – Deals in Negotiation or later stages

Ask "What's the forecast for this quarter?" to see your weighted pipeline.

---

## 12. Managing Next Steps

Every opportunity should have a clear `next_step` defined:

**Good next steps are:**
- Specific: "Send revised proposal to Anna by Friday"
- Actionable: "Schedule demo with IT team"
- Time-bound: "Follow up on pricing concerns by 11/25"

**Bad next steps:**
- Vague: "Continue discussions"
- Passive: "Waiting for customer"
- Stale: "TBD"

**Pro tip:** Update the next step after every customer interaction.
Ask "What's the next step for my biggest deal?" to surface action items.
