"""
Answer node prompt templates.

Templates for generating answers and handling company-not-found scenarios.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from backend.agent.llm.prompts import AGENT_SYSTEM_PROMPT


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
    "COMPANY_NOT_FOUND_TEMPLATE",
    "DATA_ANSWER_TEMPLATE",
]
