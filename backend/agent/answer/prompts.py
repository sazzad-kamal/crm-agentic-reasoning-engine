"""
Answer node prompt templates.

Templates for generating answers from SQL query results.
Loads system prompt from prompt.txt for clean separation.
"""

from pathlib import Path

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Load system prompt from co-located txt file
_PROMPT_PATH = Path(__file__).parent / "prompt.txt"
AGENT_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()


DATA_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(AGENT_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("""Answer the user's question using ONLY the provided data below.

User's question: {question}

{conversation_history_section}

=== CRM DATA (SQL Query Results) ===
{sql_results}

{account_context_section}

Please provide a helpful, grounded response following the rules in your system prompt.
If the data is empty or doesn't contain the answer, acknowledge this briefly."""),
    ]
)


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "DATA_ANSWER_TEMPLATE",
]
