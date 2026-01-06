"""
Follow-up node prompt templates.

Templates for generating follow-up question suggestions.
"""

from langchain_core.prompts import ChatPromptTemplate


FOLLOW_UP_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful CRM assistant. Generate 3 follow-up question suggestions."),
        (
            "human",
            """Suggest 3 follow-up questions for the user.

User's question: {question}
Current company: {company}

=== AVAILABLE DATA FOR THIS COMPANY ===
{available_data}

{conversation_history_section}

GENERATE 3 QUESTIONS:
1. First question: Drill deeper into current company's available data (use company name)
2. Second question: Another angle on current company's data (use company name)
3. Third question: Let user explore something NEW - different company, general CRM question, or documentation topic

RULES:
- Questions 1-2: ONLY ask about data types listed as available above
- Question 3: Can be general (renewals, pipeline summary) or about CRM features
- Always use company name, not "they" or "their"
- Keep questions SHORT""",
        ),
    ]
)


__all__ = [
    "FOLLOW_UP_PROMPT_TEMPLATE",
]
