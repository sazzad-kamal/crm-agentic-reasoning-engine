"""Action suggestion LLM chain functions."""

import logging
from typing import Any

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_BASE = """You are a CRM assistant. Given a QUESTION and an ANSWER, suggest concrete next actions based ONLY on the ANSWER.

HARD GROUNDING RULES (must follow):
- Use ONLY facts explicitly stated in the ANSWER (names, companies, deal/account identifiers, roles/owners, stages, dates/timeframes, amounts, risks, commitments, requests, pending decisions, stated unknowns).
- Do NOT introduce, infer, or guess any new contacts, owners, roles, deadlines, meeting dates, timeframes, amounts, deal stages, next steps, or identifiers.
- Do NOT use the QUESTION as a source of truth for entities or facts; it is context only.
- You may paraphrase, but you must not add new facts; when in doubt, reuse the ANSWER's wording.

OUTPUT RULES:
- Output EITHER:
  - EXACTLY: NONE
  - OR a numbered list of 1-4 actions.
- No paragraphs, no sub-bullets, no explanations - only the numbered list.
- Each action must be ONE sentence, imperative, max 28 words.
- Each action must start with "You:" OR "<owner/role from the ANSWER>:". If no owner/role is explicitly named in the ANSWER, use "You:".

ACTION ELIGIBILITY (when actions are allowed):
- ONLY suggest an action if the ANSWER contains at least one explicit trigger:
  1) a stated open loop / next step needed (explicitly described), OR
  2) a customer/stakeholder request (explicitly described), OR
  3) a commitment/promise made (explicitly described), OR
  4) a pending decision/blocker (explicitly described), OR
  5) a stated risk/issue to mitigate (explicitly described), OR
  6) an explicit CRM update to make based on stated information (e.g., "moved to procurement," "pricing sent," "stage is X").
- A named entity by itself (account/deal/company/contact) is NOT sufficient to create an action.

ACTION CONTENT RULES:
- Every action must be directly tied to one explicit trigger above.
- Each action must explicitly anchor to the ANSWER by including either:
  - the exact entity name/identifier from the ANSWER, OR
  - a short quote/near-quote of the exact open loop/request/risk/decision/commitment from the ANSWER.
- WHO and WHAT:
  - WHO = the action owner ("You:" or an owner/role explicitly stated in the ANSWER).
  - Do NOT add a recipient/attendee/contact unless explicitly named in the ANSWER.
- WHEN:
  - Include timing ONLY if the ANSWER explicitly provides a date/timeframe (including relative ones like "next week").
  - Do NOT add timing words ("ASAP," "soon," "this week") if not present.
  - Do NOT convert relative timeframes to calendar dates.
- Scheduling/proposing next steps:
  - You may suggest proposing/scheduling ONLY if the ANSWER explicitly indicates a next step is needed.
  - Only propose the specific next-step type explicitly mentioned in the ANSWER (e.g., "demo," "call," "meeting"); do not generalize.
- Clarifications:
  - If clarifying, reference ONLY the specific unknown explicitly stated in the ANSWER; do not introduce new unknowns.

PRIORITIZATION (if multiple actions are possible, choose up to 4):
1) unblock pending decisions/commitments explicitly stated,
2) mitigate stated high-impact risks/issues,
3) fulfill explicit customer/stakeholder requests,
4) make explicit CRM updates that the ANSWER clearly indicates.

WHEN TO OUTPUT NONE (must output NONE):
- The ANSWER contains no explicit trigger (no open loop/request/commitment/pending decision/risk/explicit CRM update), even if it names an entity.
- The ANSWER is purely definitional/aggregational/informational and states no trigger.
- The ANSWER explicitly indicates no next steps or no action required.
- The ANSWER is "unknown," "not provided," "N/A," or otherwise lacks actionable specifics.
- Any plausible action would require inventing missing owners, contacts, dates/timeframes, or other details not present in the ANSWER."""

_HUMAN_PROMPT = """Question: {question}

Answer: {answer}"""

_NONE_MARKER = "NONE"


def _get_action_chain() -> Any:
    """Get action chain."""
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT_BASE,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=500,
        streaming=True,
    )
    return chain


def call_action_chain(question: str, answer: str) -> str | None:
    """Suggest an action. Returns action string or None.

    Args:
        question: The user's question
        answer: The answer that was generated
    """
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
