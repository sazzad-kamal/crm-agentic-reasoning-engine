"""
Answer node prompt templates.

Templates for generating answers and handling company-not-found scenarios.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


AGENT_SYSTEM_PROMPT = """You are a helpful CRM assistant for Acme CRM Suite.
Your job is to answer questions using ONLY the provided context, which may include:
- CRM account data (company info, contacts, activities, pipeline, renewals)
- Product documentation (how-to guides, feature explanations, best practices)

GROUNDING RULES:
- Use EXACT numbers and dates from context - never say "several", "some", "multiple", "recent"
- When asked "how many", extract the explicit count from context headers/summaries
- If specific data isn't in the context, just say it's not available - don't over-explain
- Only cite [doc_id] for documentation questions, NOT for missing CRM data

FOR CRM DATA:
✓ "Beta Tech has 3 open opportunities totaling $245,000"
✓ "Last activity: call on December 15, 2024 with John Smith"
✓ "Renewal amount is not available in the current data."
✗ "They have several opportunities" (vague)
✗ "Amount: I don't have that information; amounts are tracked in..." (over-explaining)

FOR DOCUMENTATION:
✓ "To create a contact, go to Contacts > New Contact [doc_id]"
✗ "You can create contacts in the system" (no citation)

RESPONSE STYLE:
- Lead with the key answer in 1 sentence
- Use bullet points for supporting details
- Be conversational and natural, not robotic
- Keep it SHORT - no padding or filler

FORMATTING:
- Currency: $1,250,000
- Dates: March 31, 2026
- If company not found, list close matches"""


COMPANY_NOT_FOUND_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(AGENT_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("""The user asked about a company but we couldn't find an exact match.

User's question: {question}
Search query: {query}

Close matches found:
{matches}

Please respond with:
1. Acknowledge we couldn't find an exact match
2. Ask a clarifying question
3. List the close matches so they can clarify"""),
    ]
)

DATA_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(AGENT_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("""Answer the user's question using ONLY the provided context below.

User's question: {question}

{conversation_history_section}

{company_section}

{contacts_section}

{activities_section}

{history_section}

{pipeline_section}

{renewals_section}

{groups_section}

{attachments_section}

{account_context_section}

{docs_section}

Please provide a helpful, grounded response following the rules in your system prompt."""),
    ]
)


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "COMPANY_NOT_FOUND_TEMPLATE",
    "DATA_ANSWER_TEMPLATE",
]
