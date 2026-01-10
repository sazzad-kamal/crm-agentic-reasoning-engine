"""Prompt loading utilities.

Load and parse prompt.txt files with [system] and [human] sections.
"""

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate


def load_prompt(path: Path) -> ChatPromptTemplate:
    """Load and parse a prompt.txt file into a ChatPromptTemplate.

    The file should contain [system] and [human] section markers.

    Args:
        path: Path to the prompt.txt file

    Returns:
        ChatPromptTemplate with system and human messages
    """
    text = path.read_text(encoding="utf-8")

    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        if line.strip() == "[system]":
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "system"
            current_lines = []
        elif line.strip() == "[human]":
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "human"
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return ChatPromptTemplate.from_messages(
        [
            ("system", sections.get("system", "")),
            ("human", sections.get("human", "")),
        ]
    )


__all__ = ["load_prompt"]
