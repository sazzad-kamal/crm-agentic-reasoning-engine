# Email Marketing and Campaigns

_Last updated: 2025-11-26_

## 1. What Email Features Does Acme CRM Suite Provide?

Acme CRM Suite includes light email marketing features focused on:

- communicating with existing customers and warm leads,
- onboarding new customers,
- sending announcements and updates.

It is not intended to replace dedicated, high‑volume marketing platforms for
cold outreach.

Supported functions:

- One‑off email campaigns to contact groups,
- Simple multi‑step sequences (e.g., onboarding),
- Tracking open and click metrics,
- Managing unsubscribes and basic compliance.

## 2. Contact Lists and Groups

You build email targets from:

- **Contact filters** (e.g., contacts with a certain lifecycle stage),
- **Company fields** (e.g., Pro customers in Europe),
- **Activity criteria** (e.g., no meaningful activity in 60 days).

Once you define filters, you can save them as **Groups** and reuse them
across campaigns.

Examples:

- All active customers with renewals in the next 90 days.
- All admins or champions at Pro plan accounts.
- All trial accounts created in the last 14 days.

## 3. Campaign Types

### 3.1 Broadcast Campaigns

Single send to a group of contacts.

You choose:

- Subject and preview text,
- Email template and content,
- Sender name and address,
- Send time (immediate or scheduled).

Use for:

- product announcements,
- policy changes,
- maintenance notices.

### 3.2 Sequences

Sequences are multi‑email flows triggered by an event or enrollment.

Examples:

- **Onboarding series** – welcome email, setup tips, feature highlights.
- **Renewal reminders** – reminders 60, 30, and 7 days before renewal.
- **Trial nurture** – 3–5 emails over the trial period.

Each step has:

- delay (e.g., send 3 days after previous),
- email content,
- optional exit rules (e.g., stop if contact becomes a paying customer).

## 4. Deliverability Basics

For small businesses, protecting your sending reputation is important:

- Avoid sending bulk campaigns to very old, unengaged lists.
- Make sure recipients have given consent.
- Use clear unsubscribe links (required).

Your CRM tracks:

- deliveries,
- bounces (temporary vs permanent),
- unsubscribes.

If your bounce or complaint rates are high, you may see:

- suggestions to clean your list,
- temporary limits on sending.

## 5. Templates and Personalization

### Built-in Email Templates

Acme CRM Suite includes ready-to-use templates:

**Onboarding Templates:**
- Welcome Email – greet new customers with getting-started links
- Setup Checklist – guide users through initial configuration
- Feature Highlight – introduce key features over the first week

**Renewal Templates:**
- Renewal Reminder (60 days) – early notice with account summary
- Renewal Reminder (30 days) – detailed renewal information
- Renewal Reminder (7 days) – urgent final reminder
- Thank You for Renewing – confirmation and next steps

**Engagement Templates:**
- Check-in Email – "How's everything going?" for quiet accounts
- Product Update – announce new features
- Feedback Request – ask for reviews or testimonials
- Re-engagement – win back inactive users

**Sales Templates:**
- Meeting Follow-up – summary and next steps after a call
- Proposal Sent – accompany a quote or proposal
- Deal Won – thank you and onboarding handoff

Templates can be customized or you can create your own from scratch.

### Merge Fields (Personalization)

You can use merge fields such as:

- `{{contact.first_name}}`
- `{{contact.last_name}}`
- `{{company.name}}`
- `{{company.plan}}`
- `{{company.renewal_date}}`
- `{{opportunity.value}}`
- `{{owner.name}}` (account owner)

Good practices:

- keep templates simple and easy to scan,
- use plain, clear language,
- include a strong call to action (e.g., "Book a call", "Review your usage").

## 6. Metrics and Follow‑up

For each campaign or sequence, review:

- open rate,
- click‑through rate,
- unsubscribe rate,
- bounce rate.

Use these to:

- identify segments that respond well,
- adjust subject lines and content,
- decide which contacts should receive follow‑up calls.

If a contact engages but doesn’t take action, consider creating a **Task**
for the account owner to follow up by phone.

For details on reporting metrics, see:
`reports_dashboards_and_analytics.md`.

## 7. Common Questions

- **“Can I send cold outreach from Acme CRM Suite?”**  
  It’s designed for existing customers and warm leads. For large‑scale cold
  outreach, use a dedicated tool that supports that use case and
  compliance.

- **“How do I avoid sending too many emails?”**  
  Use groups and activity‑based filters (e.g., “opened at least one email
  in the last 90 days”), and respect unsubscribes.

- **“Where can I see which campaigns a contact received?”**  
  Open the contact’s history; email campaign sends and key events typically
  appear as activities.
