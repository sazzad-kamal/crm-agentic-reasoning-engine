"""
Follow-up node prompt templates.

Templates for generating follow-up question suggestions.
Loads prompt from prompt.txt for clean separation.
"""

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

# Load and parse prompt from co-located txt file
_PROMPT_PATH = Path(__file__).parent / "prompt.txt"
_PROMPT_TEXT = _PROMPT_PATH.read_text(encoding="utf-8")

# Parse [system] and [human] sections
_sections: dict[str, str] = {}
_current_section: str | None = None
_current_lines: list[str] = []

for line in _PROMPT_TEXT.split("\n"):
    if line.strip() == "[system]":
        if _current_section:
            _sections[_current_section] = "\n".join(_current_lines).strip()
        _current_section = "system"
        _current_lines = []
    elif line.strip() == "[human]":
        if _current_section:
            _sections[_current_section] = "\n".join(_current_lines).strip()
        _current_section = "human"
        _current_lines = []
    else:
        _current_lines.append(line)

if _current_section:
    _sections[_current_section] = "\n".join(_current_lines).strip()

FOLLOW_UP_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _sections.get("system", "")),
        ("human", _sections.get("human", "")),
    ]
)


__all__ = [
    "FOLLOW_UP_PROMPT_TEMPLATE",
]
