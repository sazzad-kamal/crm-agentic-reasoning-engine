# History, Activities, and Calendar

_Last updated: 2025-11-26_

## 1. History vs Activities

Acme CRM Suite keeps a **history** of what has happened with each company
and contact. Depending on configuration, this may appear as:

- a unified **History** tab,
- or separate **Activities** and **Notes** sections.

Typical history items:

- Logged calls,
- Meetings,
- Emails (subject + key details, not full email body),
- Notes (free‑form text),
- System events (plan changes, user invites, etc.).

An **Activity** is usually a record with:

- `Type` (Call, Meeting, Email, Note),
- `Date/Time`,
- `Owner`,
- Subject & optional description,
- Linked company/contact/opportunity.

## 2. Logging Calls, Meetings, and Emails

You can log interactions in a few ways:

- Manually from a company or contact page (“Log Call”, “Add Note”),
- From a calendar event (if calendar sync is enabled),
- From an email plugin or BCC address (if configured).

Best practices:

- Use clear, short subjects (e.g., “Kickoff call – discussed integration
  timeline”).
- Record important decisions and follow‑ups in the description.
- Link the activity to the most relevant company and contact so it is
  visible in the right history.

## 3. Tasks and Reminders

Tasks represent future work you don’t want to forget:

- follow‑up calls,
- demos,
- trial check‑ins,
- renewal conversations.

Common fields:

- `Subject`
- `Due Date`
- `Priority`
- `Owner`
- `Status` (Open, Completed, Cancelled)
- Linked company/contact/opportunity

You can view tasks from:

- “My Tasks” list,
- the calendar (if enabled),
- company/contact/opportunity pages.

## 4. Calendar Integration (Optional)

Some deployments allow Acme CRM Suite to connect to your email/calendar
provider. Depending on setup, this can:

- show CRM tasks on your calendar,
- sync meetings as activities,
- help you log time spent on each account.

Check with your administrator if calendar integration is available in your
plan and region.

## 5. Activity Views and Filters

Common history/activity list filters:

- Date range (e.g., last 7, 30, 90 days),
- Owner,
- Activity type (Calls only, Meetings only),
- Company or segment.

Useful views for SMBs:

- “My Activities – Last 30 Days” – see what you’ve done.
- “No Activity – 30 Days” – find companies that have gone quiet.
- “Upcoming Calls” – tasks due in the next week.

## 6. Account Health and At-Risk Flags

Acme CRM Suite automatically calculates **account health** based on
engagement patterns. This helps identify accounts that need attention.

### Health Status Levels

- **Healthy** (Green) – Regular engagement, no concerns
- **Needs Attention** (Yellow) – Some warning signs present
- **At-Risk** (Red) – Significant concerns, action needed

### What Causes an Account to Be Marked At-Risk?

Accounts are flagged as at-risk based on several factors:

1. **Low activity** – No calls, meetings, or emails logged in 30+ days
2. **Missed check-ins** – Scheduled tasks or calls that weren't completed
3. **Declining engagement** – Reduced response rates or meeting frequency
4. **Support issues** – Open tickets or unresolved complaints
5. **Upcoming renewal without engagement** – Renewal within 60 days but
   no recent contact
6. **Contact changes** – Key contacts leaving or changing roles

### Low-Activity Warnings

A low-activity warning is triggered when:

- No activities have been logged for 30 days (default threshold)
- The account is marked as "Active" status
- There are no scheduled future tasks or meetings

## 7. Troubleshooting Tips (for SMB users)

If you can’t find an activity:

- Confirm you are looking at the **correct company/contact**.
- Check date filters (e.g., it might be outside the current range).
- Confirm that the integration or email logging feature was enabled at the
  time of the interaction.

If history looks cluttered:

- Use type filters (e.g., hide system events, show only calls/meetings).
- Encourage your team to use short subjects and avoid duplicate entries.

For strategic use of activities with sales and success, see:
`opportunities_pipeline_and_forecasts.md`.

## 8. Common Questions

- **“How can I see everything that happened with a single account?”**  
  Open the Company record and use the History or Activities tab; filter by
  date range if needed.

- **“What’s the difference between a Task and an Activity?”**  
  Tasks represent future work you plan to do. Activities are things that
  already happened. Completing a Task may create or link to an Activity.

- **“How do I find accounts with no recent activity?”**  
  Use an Activity report or dashboard tile such as “Accounts with no
  activity in the last 30 days”, or create a saved view that filters on
  activity date.
