"""Evaluate email feature quality using GPT-5.2-pro.

Two evaluation types:
1. Category Classification: Are the right contacts selected with good reasons?
2. Email Generation: Are the generated emails high quality?

Run with: python -m backend.eval.act.email_scorer
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

load_dotenv()

from backend.agent.email.generator import (  # noqa: E402, I001
    CATEGORY_DESCRIPTIONS,
    EMAIL_QUESTIONS,
    generate_email,
    get_contacts_for_category,
)
from backend.core.llm import create_openai_chain  # noqa: E402


# =============================================================================
# Scoring Models
# =============================================================================

class CategoryScore(BaseModel):
    """Scores for category classification results."""

    relevance: int = Field(ge=0, le=5, description="Do the contacts actually fit this category? 0=wrong contacts, 5=perfect matches")
    reason_quality: int = Field(ge=0, le=5, description="Are the AI-generated reasons specific and helpful? 0=vague/wrong, 5=excellent")
    contact_selection: int = Field(ge=0, le=5, description="Are these the RIGHT contacts to prioritize? 0=poor selection, 5=excellent prioritization")
    actionability: int = Field(ge=0, le=5, description="Can Ken take action based on this list? 0=useless, 5=immediately actionable")

    relevance_reason: str = Field(description="Brief reason for relevance score")
    reason_quality_reason: str = Field(description="Brief reason for reason_quality score")
    contact_selection_reason: str = Field(description="Brief reason for contact_selection score")
    actionability_reason: str = Field(description="Brief reason for actionability score")

    overall_assessment: str = Field(description="Overall assessment of this category's results")


class EmailScore(BaseModel):
    """Scores for generated email quality."""

    personalization: int = Field(ge=0, le=5, description="Does the email reference specific context? 0=generic, 5=highly personalized")
    professionalism: int = Field(ge=0, le=5, description="Is the tone professional but friendly? 0=inappropriate, 5=perfect tone")
    clarity: int = Field(ge=0, le=5, description="Is the email clear and well-written? 0=confusing, 5=crystal clear")
    call_to_action: int = Field(ge=0, le=5, description="Is there a clear next step? 0=no CTA, 5=clear actionable CTA")

    personalization_reason: str = Field(description="Brief reason for personalization score")
    professionalism_reason: str = Field(description="Brief reason for professionalism score")
    clarity_reason: str = Field(description="Brief reason for clarity score")
    call_to_action_reason: str = Field(description="Brief reason for call_to_action score")

    would_send: bool = Field(description="Would you actually send this email? True/False")
    improvement_suggestion: str = Field(description="What would make this email better?")


# =============================================================================
# Prompts
# =============================================================================

_CATEGORY_SYSTEM_PROMPT = """You are evaluating a CRM contact classification system for a sales manager named Ken.

The system analyzes CRM history and identifies contacts needing follow-up in specific categories.
Score each dimension 0-5:

- **Relevance**: Do the returned contacts actually fit the category description?
- **Reason Quality**: Are the AI-generated reasons specific, accurate, and helpful?
- **Contact Selection**: Given the category, are these the right contacts to prioritize?
- **Actionability**: Can Ken look at this list and immediately know what to do?

Be critical. A sales manager needs accurate, actionable results - not false positives or vague reasons."""

_CATEGORY_HUMAN_PROMPT = """## Category
**{category_id}**: {category_description}

## Returned Contacts ({count} total)
```json
{contacts_json}
```

Evaluate this classification result. Are these the right contacts with good reasons?"""


_EMAIL_SYSTEM_PROMPT = """You are evaluating AI-generated follow-up emails for a sales manager named Ken.

Score each dimension 0-5:

- **Personalization**: Does the email reference the specific contact, company, and interaction context?
- **Professionalism**: Is the tone professional but friendly? No errors or awkward phrasing?
- **Clarity**: Is the email clear, concise, and easy to understand?
- **Call to Action**: Is there a clear next step for the recipient?

Be critical. This email represents Ken's professional reputation."""

_EMAIL_HUMAN_PROMPT = """## Contact
Name: {contact_name}
Company: {company}
Category: {category} - {category_description}

## Generated Email
**Subject:** {subject}

**Body:**
{body}

Evaluate this email. Would Ken be comfortable sending it?"""


# =============================================================================
# Scoring Functions
# =============================================================================

def score_category(
    category_id: str,
    category_description: str,
    contacts: list[dict],
) -> CategoryScore:
    """Score a category classification result using GPT-5.2-pro."""

    chain = create_openai_chain(
        system_prompt=_CATEGORY_SYSTEM_PROMPT,
        human_prompt=_CATEGORY_HUMAN_PROMPT,
        max_tokens=1024,
        structured_output=CategoryScore,
        streaming=False,
        model="gpt-5.2-pro",
    )

    # Truncate contacts if too many
    contacts_for_eval = contacts[:10]
    contacts_json = json.dumps(contacts_for_eval, indent=2, default=str)

    result: CategoryScore = chain.invoke({
        "category_id": category_id,
        "category_description": category_description,
        "count": len(contacts),
        "contacts_json": contacts_json,
    })

    return result


def score_email(
    contact_name: str,
    company: str,
    category: str,
    category_description: str,
    subject: str,
    body: str,
) -> EmailScore:
    """Score a generated email using GPT-5.2-pro."""

    chain = create_openai_chain(
        system_prompt=_EMAIL_SYSTEM_PROMPT,
        human_prompt=_EMAIL_HUMAN_PROMPT,
        max_tokens=1024,
        structured_output=EmailScore,
        streaming=False,
        model="gpt-5.2-pro",
    )

    result: EmailScore = chain.invoke({
        "contact_name": contact_name,
        "company": company or "Unknown",
        "category": category,
        "category_description": category_description,
        "subject": subject,
        "body": body,
    })

    return result


# =============================================================================
# Report Dataclasses
# =============================================================================

@dataclass
class CategoryResult:
    """Result for one category evaluation."""
    category_id: str
    category_description: str
    contact_count: int
    contacts: list[dict]
    scores: dict
    total_score: int


@dataclass
class EmailResult:
    """Result for one email evaluation."""
    contact_id: str
    contact_name: str
    company: str
    category: str
    subject: str
    body: str
    scores: dict
    total_score: int


@dataclass
class EvaluationReport:
    """Full evaluation report."""
    timestamp: str
    category_results: list[CategoryResult] = field(default_factory=list)
    email_results: list[EmailResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# =============================================================================
# Main Evaluation
# =============================================================================

async def run_evaluation() -> EvaluationReport:
    """Run full email feature evaluation."""

    report = EvaluationReport(timestamp=datetime.utcnow().isoformat())

    print("=" * 70)
    print("Email Feature Evaluation")
    print("Using: GPT-5.2-pro as judge")
    print("=" * 70)

    # ===================
    # Part 1: Category Classification (5 cases)
    # ===================
    print("\n" + "=" * 70)
    print("PART 1: Category Classification Evaluation (5 categories)")
    print("=" * 70)

    category_totals = {"relevance": 0, "reason_quality": 0, "contact_selection": 0, "actionability": 0}

    for i, q in enumerate(EMAIL_QUESTIONS, 1):
        category_id = q["id"]
        category_desc = CATEGORY_DESCRIPTIONS[category_id]

        print(f"\n[{i}/5] {q['label']}")
        print("-" * 50)

        # Fetch contacts for this category
        try:
            contacts = await get_contacts_for_category(category_id)
            print(f"  Found {len(contacts)} contacts")

            if not contacts:
                print("  SKIPPED: No contacts found")
                continue

            # Score with LLM
            scores = score_category(category_id, category_desc, contacts)
            total = scores.relevance + scores.reason_quality + scores.contact_selection + scores.actionability

            result = CategoryResult(
                category_id=category_id,
                category_description=category_desc,
                contact_count=len(contacts),
                contacts=contacts,
                scores=scores.model_dump(),
                total_score=total,
            )
            report.category_results.append(result)

            # Accumulate
            category_totals["relevance"] += scores.relevance
            category_totals["reason_quality"] += scores.reason_quality
            category_totals["contact_selection"] += scores.contact_selection
            category_totals["actionability"] += scores.actionability

            # Print
            status = "PASS" if total >= 12 else "FAIL"
            print(f"  [{status}] Total: {total}/20")
            print(f"    Relevance:        {scores.relevance}/5 - {scores.relevance_reason}")
            print(f"    Reason Quality:   {scores.reason_quality}/5 - {scores.reason_quality_reason}")
            print(f"    Contact Selection:{scores.contact_selection}/5 - {scores.contact_selection_reason}")
            print(f"    Actionability:    {scores.actionability}/5 - {scores.actionability_reason}")
            print(f"  Assessment: {scores.overall_assessment}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # ===================
    # Part 2: Email Generation (1 email per category)
    # ===================
    print("\n" + "=" * 70)
    print("PART 2: Email Generation Evaluation (1 per category)")
    print("=" * 70)

    email_totals = {"personalization": 0, "professionalism": 0, "clarity": 0, "call_to_action": 0}
    emails_evaluated = 0

    for i, cat_result in enumerate(report.category_results, 1):
        if not cat_result.contacts:
            continue

        # Pick first contact for email generation
        contact = cat_result.contacts[0]
        contact_id = contact.get("contactId")
        contact_name = contact.get("name", "Unknown")
        company = contact.get("company", "")

        print(f"\n[{i}/5] Generating email for: {contact_name}")
        print(f"  Category: {cat_result.category_id}")
        print("-" * 50)

        try:
            # Generate email
            email_data = await generate_email(contact_id, cat_result.category_id)

            subject = email_data["subject"]
            body = email_data["body"]

            print(f"  Subject: {subject}")
            print(f"  Body preview: {body[:100]}...")

            # Score with LLM
            scores = score_email(
                contact_name=contact_name,
                company=company,
                category=cat_result.category_id,
                category_description=cat_result.category_description,
                subject=subject,
                body=body,
            )
            total = scores.personalization + scores.professionalism + scores.clarity + scores.call_to_action

            result = EmailResult(
                contact_id=contact_id,
                contact_name=contact_name,
                company=company,
                category=cat_result.category_id,
                subject=subject,
                body=body,
                scores=scores.model_dump(),
                total_score=total,
            )
            report.email_results.append(result)

            # Accumulate
            email_totals["personalization"] += scores.personalization
            email_totals["professionalism"] += scores.professionalism
            email_totals["clarity"] += scores.clarity
            email_totals["call_to_action"] += scores.call_to_action
            emails_evaluated += 1

            # Print
            status = "PASS" if total >= 12 else "FAIL"
            would_send = "✓ Yes" if scores.would_send else "✗ No"
            print(f"  [{status}] Total: {total}/20")
            print(f"    Personalization:  {scores.personalization}/5 - {scores.personalization_reason}")
            print(f"    Professionalism:  {scores.professionalism}/5 - {scores.professionalism_reason}")
            print(f"    Clarity:          {scores.clarity}/5 - {scores.clarity_reason}")
            print(f"    Call to Action:   {scores.call_to_action}/5 - {scores.call_to_action_reason}")
            print(f"  Would Send: {would_send}")
            print(f"  Improvement: {scores.improvement_suggestion}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # ===================
    # Summary
    # ===================
    n_cat = len(report.category_results) or 1
    n_email = emails_evaluated or 1

    report.summary = {
        "categories": {
            "count": len(report.category_results),
            "avg_relevance": category_totals["relevance"] / n_cat,
            "avg_reason_quality": category_totals["reason_quality"] / n_cat,
            "avg_contact_selection": category_totals["contact_selection"] / n_cat,
            "avg_actionability": category_totals["actionability"] / n_cat,
            "avg_total": sum(category_totals.values()) / n_cat,
            "pass_count": sum(1 for r in report.category_results if r.total_score >= 12),
        },
        "emails": {
            "count": emails_evaluated,
            "avg_personalization": email_totals["personalization"] / n_email,
            "avg_professionalism": email_totals["professionalism"] / n_email,
            "avg_clarity": email_totals["clarity"] / n_email,
            "avg_call_to_action": email_totals["call_to_action"] / n_email,
            "avg_total": sum(email_totals.values()) / n_email,
            "pass_count": sum(1 for r in report.email_results if r.total_score >= 12),
            "would_send_count": sum(1 for r in report.email_results if r.scores.get("would_send")),
        },
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nCategory Classification:")
    print(f"  Avg Relevance:         {report.summary['categories']['avg_relevance']:.1f}/5")
    print(f"  Avg Reason Quality:    {report.summary['categories']['avg_reason_quality']:.1f}/5")
    print(f"  Avg Contact Selection: {report.summary['categories']['avg_contact_selection']:.1f}/5")
    print(f"  Avg Actionability:     {report.summary['categories']['avg_actionability']:.1f}/5")
    print(f"  Avg Total:             {report.summary['categories']['avg_total']:.1f}/20")
    print(f"  Pass/Fail:             {report.summary['categories']['pass_count']}/{len(report.category_results) - report.summary['categories']['pass_count']}")

    print("\nEmail Generation:")
    print(f"  Avg Personalization:   {report.summary['emails']['avg_personalization']:.1f}/5")
    print(f"  Avg Professionalism:   {report.summary['emails']['avg_professionalism']:.1f}/5")
    print(f"  Avg Clarity:           {report.summary['emails']['avg_clarity']:.1f}/5")
    print(f"  Avg Call to Action:    {report.summary['emails']['avg_call_to_action']:.1f}/5")
    print(f"  Avg Total:             {report.summary['emails']['avg_total']:.1f}/20")
    print(f"  Pass/Fail:             {report.summary['emails']['pass_count']}/{emails_evaluated - report.summary['emails']['pass_count']}")
    print(f"  Would Send:            {report.summary['emails']['would_send_count']}/{emails_evaluated}")

    return report


def save_report(report: EvaluationReport, output_path: Path) -> None:
    """Save evaluation report to JSON."""
    data = {
        "timestamp": report.timestamp,
        "summary": report.summary,
        "category_results": [asdict(r) for r in report.category_results],
        "email_results": [asdict(r) for r in report.email_results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")


def main() -> None:
    """Run email feature evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate email feature quality using GPT-5.2-pro")
    parser.add_argument("--output", "-o", type=Path, help="Output path (default: auto-generated)")
    args = parser.parse_args()

    # Run async evaluation
    report = asyncio.run(run_evaluation())

    # Save report
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"email_eval_{timestamp}.json"

    save_report(report, output_path)


if __name__ == "__main__":
    main()
