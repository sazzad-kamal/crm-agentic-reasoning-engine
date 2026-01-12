"""Slot-based query planner for LLM-driven SQL generation."""

import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import jsonschema
from openai import OpenAI

from backend.agent.core.config import get_config
from backend.agent.route.slot_query import SlotPlan, SlotQuery
from backend.utils.prompt import load_prompt

logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent


@lru_cache
def _load_schema() -> dict:
    """Load JSON schema (cached)."""
    with open(_DIR / "schema.json") as f:
        return json.load(f)


@lru_cache
def _load_examples() -> str:
    """Load and validate examples, return formatted string for prompt."""
    with open(_DIR / "examples.json") as f:
        examples = json.load(f)

    schema = _load_schema()

    # Validate each example against the schema
    for ex in examples:
        jsonschema.validate(ex["output"], schema)

    # Format as prompt string
    lines = []
    for ex in examples:
        lines.append(f'User: "{ex["question"]}"')
        lines.append(json.dumps(ex["output"], separators=(",", ":")))
        lines.append("")

    return "\n".join(lines).strip()


@lru_cache
def _get_client() -> OpenAI:
    """Get OpenAI client (cached)."""
    return OpenAI()


def get_slot_plan(question: str, conversation_history: str = "") -> SlotPlan:
    """
    Get slot-based query plan from LLM.

    LLM outputs structured slots (table, filters, order_by) which are
    converted to SQL programmatically - more reliable than raw SQL generation.
    """
    config = get_config()

    prompt = load_prompt(_DIR / "prompt.txt").format(
        today=datetime.now().strftime("%Y-%m-%d"),
        examples=_load_examples(),
        conversation_history=conversation_history or "",
        question=question,
    )

    response = _get_client().chat.completions.create(
        model=config.router_model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "SlotPlan", "strict": True, "schema": _load_schema()},
        },
    )

    content = response.choices[0].message.content
    if not content:
        return SlotPlan(queries=[], needs_rag=False)

    data = json.loads(content)
    result = SlotPlan(
        queries=[SlotQuery(**q) for q in data.get("queries", [])],
        needs_rag=data.get("needs_rag", False),
    )

    logger.info("Slot Planner: %d queries, needs_rag=%s", len(result.queries), result.needs_rag)
    return result
