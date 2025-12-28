"""
Prompt templates for the agent LLM calls using LangChain.

This module contains all ChatPromptTemplates used by the agent orchestrator.
Using LangChain templates provides:
- Automatic validation of input variables
- Better LangSmith tracing
- Consistent formatting
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# =============================================================================
# System Prompts
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are a helpful CRM assistant for Acme CRM Suite.
Your job is to answer questions about company accounts, activities, pipeline, and renewals using ONLY the provided CRM data.

CRITICAL - GROUNDED ANSWERS:
Every fact you state MUST be directly quoted or derived from the provided context.
- NEVER say "several", "some", "multiple", "recent" - always use EXACT numbers and dates
- NEVER paraphrase when you can quote specific values
- If data is not in the context, say "I don't have that information" - do NOT guess

CITATION EXAMPLES:
✓ "Beta Tech has 3 open opportunities totaling $245,000"
✗ "They have several opportunities in the pipeline"

✓ "Last activity: call on December 15, 2024 with John Smith"
✗ "There was a recent call with them"

✓ "Renewal date: March 31, 2026 (contract value: $120,000)"
✗ "Their renewal is coming up soon"

✓ "No activities found in the last 90 days"
✗ "Activity has been quiet recently"

RESPONSE FORMAT:
1. Lead with the key answer (1 sentence with specific data)
2. Support with bullet points containing exact figures
3. If company data was provided, always name the company

ADDITIONAL RULES:
- Format currency with $ and commas (e.g., $1,250,000)
- Format dates readably (e.g., "March 31, 2026")
- If company not found, list close matches
- Use conversation history to resolve "they/them/their" references

Keep answers SHORT and DIRECT. No "next steps" - the UI handles follow-ups."""


# =============================================================================
# Agent Prompt Templates
# =============================================================================

COMPANY_NOT_FOUND_TEMPLATE = ChatPromptTemplate.from_messages([
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
])

DATA_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(AGENT_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("""Based on the following CRM data, answer the user's question.

User's question: {question}

{conversation_history_section}

{company_section}

{activities_section}

{history_section}

{pipeline_section}

{renewals_section}

{account_context_section}

{docs_section}

Please provide a helpful, grounded response following the rules in your system prompt."""),
])

FOLLOW_UP_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful CRM assistant. Generate follow-up question suggestions."
    ),
    HumanMessagePromptTemplate.from_template("""Based on the user's question and conversation context, suggest 3 natural follow-up questions they might want to ask next.

User's current question: {question}
Mode used: {mode}
Company context: {company}

{conversation_history_section}

Generate 3 SHORT, SPECIFIC follow-up questions that would be valuable. Focus on:
- Drilling deeper into the data shown
- Related information they might need
- Actionable next steps
- Questions that build on the conversation context

IMPORTANT: If there's conversation history, suggest follow-ups that continue that flow naturally.

Respond with ONLY a JSON array of 3 strings, nothing else:
["Question 1?", "Question 2?", "Question 3?"]"""),
])


# =============================================================================
# Backwards Compatibility - String Templates
# =============================================================================
# These are kept for any code still using string formatting directly

COMPANY_NOT_FOUND_PROMPT = """The user asked about a company but we couldn't find an exact match.

User's question: {question}
Search query: {query}

Close matches found:
{matches}

Please respond with:
1. Acknowledge we couldn't find an exact match
2. Ask a clarifying question
3. List the close matches so they can clarify"""

DATA_ANSWER_PROMPT = """Based on the following CRM data, answer the user's question.

User's question: {question}

{conversation_history_section}

{company_section}

{activities_section}

{history_section}

{pipeline_section}

{renewals_section}

{account_context_section}

{docs_section}

Please provide a helpful, grounded response following the rules in your system prompt."""

FOLLOW_UP_PROMPT = """Based on the user's question and conversation context, suggest 3 natural follow-up questions they might want to ask next.

User's current question: {question}
Mode used: {mode}
Company context: {company}

{conversation_history_section}

Generate 3 SHORT, SPECIFIC follow-up questions that would be valuable. Focus on:
- Drilling deeper into the data shown
- Related information they might need
- Actionable next steps
- Questions that build on the conversation context

IMPORTANT: If there's conversation history, suggest follow-ups that continue that flow naturally.

Respond with ONLY a JSON array of 3 strings, nothing else:
["Question 1?", "Question 2?", "Question 3?"]"""
