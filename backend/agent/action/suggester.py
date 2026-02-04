"""Action suggestion LLM chain functions."""

import logging
import os
from typing import Any

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)

# Check for demo mode
_DEMO_MODE = os.getenv("ACME_DEMO_MODE", "").lower() in ("true", "1")

_SYSTEM_PROMPT_BASE = """You are a CRM assistant. Given a question and answer, suggest next actions.

RULES:
- Output ONLY a numbered list (2-4 items), one action per line
- Each action: one short sentence (max 20 words), specific (who + what + when)
- Reference entity names from the answer
- NO paragraphs, NO sub-bullets, NO explanations — just the numbered list
- If no action is appropriate (simple lookups, counts, aggregations), respond with exactly: NONE

GOOD example:
1. Schedule renewal call with Sarah Chen at Beta Tech by Feb 5
2. Prepare pricing comparison for the Enterprise upgrade proposal
3. Flag Delta Health renewal as at-risk and assign to account manager

BAD example (too verbose):
This week: Prioritize the two Proposal-stage deals and schedule close-plan calls with stakeholders..."""

# Extended prompt for Act! demo mode with CRM-specific actions
_ACT_ACTION_CONTEXT = """

## Act! CRM Action Context
When suggesting actions for Act! CRM users, focus on:

### Follow-up Actions
- Schedule calls/meetings with specific contacts
- Send follow-up emails referencing past conversations
- Create history entries to document interactions
- Update contact last-reach dates

### Opportunity Actions
- Update opportunity stage or probability
- Add notes about deal progress
- Schedule next-step activities
- Link additional contacts to deals

### Pipeline Management
- Flag stale opportunities for review
- Schedule close-plan meetings for deals closing soon
- Update estimated close dates if slipping
- Document competitor information"""


def _get_system_prompt() -> str:
    """Get system prompt with optional Act! action context."""
    if _DEMO_MODE:
        return _SYSTEM_PROMPT_BASE + _ACT_ACTION_CONTEXT
    return _SYSTEM_PROMPT_BASE

_HUMAN_PROMPT = """Question: {question}

Answer: {answer}"""

_NONE_MARKER = "NONE"


def _get_action_chain() -> Any:
    """Get action chain with optional Act! context."""
    chain = create_openai_chain(
        system_prompt=_get_system_prompt(),
        human_prompt=_HUMAN_PROMPT,
        max_tokens=500,
        streaming=True,
    )
    logger.debug("Created action chain (demo_mode=%s)", _DEMO_MODE)
    return chain


def call_action_chain(question: str, answer: str, guidance: str = "") -> str | None:
    """Suggest an action. Returns action string or None.

    Args:
        question: The user's question
        answer: The answer that was generated
        guidance: Optional guidance for what kind of action to suggest
    """
    # If guidance provided, append it to the question
    question_with_guidance = f"{question}\n\n[Action guidance: {guidance}]" if guidance else question

    result: str = _get_action_chain().invoke({
        "question": question_with_guidance,
        "answer": answer,
    })
    action = result.strip()

    if not action or action.upper() == _NONE_MARKER:
        logger.debug("No action suggested")
        return None

    return action


__all__ = ["call_action_chain"]
