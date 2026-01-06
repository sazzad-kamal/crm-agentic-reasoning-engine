"""
LLM interaction module.

Contains helper functions and prompt templates for LLM-based operations.

Note: Router functions have moved to backend.agent.route.router
"""

# Re-export router functions for backward compatibility
from backend.agent.route.router import (
    route_question,
    llm_route_question,
    detect_owner_from_starter,
    LLMRouterError,
)
from backend.agent.llm.helpers import (
    call_docs_rag,
    call_account_rag,
    generate_follow_up_suggestions,
    call_answer_chain,
    call_not_found_chain,
    FollowUpSuggestions,
)
from backend.agent.llm.prompts import AGENT_SYSTEM_PROMPT
# Re-export node-specific prompts for backward compatibility
from backend.agent.answer.prompts import COMPANY_NOT_FOUND_TEMPLATE, DATA_ANSWER_TEMPLATE
from backend.agent.followup.prompts import FOLLOW_UP_PROMPT_TEMPLATE
from backend.agent.route.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_EXAMPLES, ROUTER_PROMPT_TEMPLATE

__all__ = [
    # Router (re-exported from route/)
    "route_question",
    "llm_route_question",
    "detect_owner_from_starter",
    "LLMRouterError",
    # Helpers
    "call_docs_rag",
    "call_account_rag",
    "generate_follow_up_suggestions",
    "call_answer_chain",
    "call_not_found_chain",
    "FollowUpSuggestions",
    # Prompts
    "AGENT_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "ROUTER_EXAMPLES",
    "ROUTER_PROMPT_TEMPLATE",
    "COMPANY_NOT_FOUND_TEMPLATE",
    "DATA_ANSWER_TEMPLATE",
    "FOLLOW_UP_PROMPT_TEMPLATE",
]
