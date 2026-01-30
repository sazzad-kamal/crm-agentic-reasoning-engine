"""Action suggestion LLM chain functions."""

import logging
from functools import cache
from typing import Any

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a CRM assistant. Given a question and answer, suggest next actions.

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

_HUMAN_PROMPT = """Question: {question}

Answer: {answer}"""

_NONE_MARKER = "NONE"


@cache
def _get_action_chain() -> Any:
    """Get cached action chain."""
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=500,
        streaming=True,
    )
    logger.debug("Created action chain")
    return chain


def call_action_chain(question: str, answer: str) -> str | None:
    """Suggest an action. Returns action string or None."""
    result: str = _get_action_chain().invoke({
        "question": question,
        "answer": answer,
    })
    action = result.strip()

    if not action or action.upper() == _NONE_MARKER:
        logger.debug("No action suggested")
        return None

    return action


__all__ = ["call_action_chain"]
