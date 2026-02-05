"""Iterate with GPT-5.2-pro to improve system prompts to 10/10.

Asks GPT-5.2-pro why it gave specific scores and gets recommendations
for improvement, then generates improved versions.

Run with: python -m backend.eval.act.prompt_improver
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from backend.agent.action.suggester import _SYSTEM_PROMPT_BASE as ACTION_PROMPT

# Import prompts from actual codebase files
from backend.agent.answer.answerer import _SYSTEM_PROMPT_BASE as ANSWER_PROMPT
from backend.core.llm import create_openai_chain


class DetailedScore(BaseModel):
    """Detailed explanation for a single scoring dimension."""

    score: int = Field(ge=0, le=10, description="Score out of 10")
    current_issue: str = Field(description="What's wrong with the current prompt for this dimension")
    specific_fix: str = Field(description="Exact wording change or addition to fix this")
    after_fix_score: int = Field(ge=0, le=10, description="Expected score after applying the fix")


class AnswerPromptAnalysis(BaseModel):
    """GPT-5.2-pro's analysis of the ANSWER prompt."""

    grounding: DetailedScore = Field(description="How well does it prevent hallucination? Must use ONLY provided data.")
    clarity: DetailedScore = Field(description="How clear and unambiguous are the instructions?")
    completeness: DetailedScore = Field(description="Does it handle empty data, partial data, stale data, ambiguous questions?")
    robustness: DetailedScore = Field(description="How well does it resist prompt injection and handle malformed input?")

    overall_score: int = Field(ge=0, le=10, description="Overall effectiveness score")
    top_priority_fix: str = Field(description="The single most impactful change to make first")
    improved_prompt: str = Field(description="Complete rewritten prompt that would score 10/10")


class ActionPromptAnalysis(BaseModel):
    """GPT-5.2-pro's analysis of the ACTION prompt."""

    grounding: DetailedScore = Field(description="Does it ONLY reference entities/facts from the answer? No fabricated names/dates.")
    actionability: DetailedScore = Field(description="Are the suggested actions useful, concrete, and actually doable?")
    specificity: DetailedScore = Field(description="Does it enforce clear who/what/when without fabricating details?")
    robustness: DetailedScore = Field(description="Does it correctly identify when to say NONE? Handle edge cases?")

    overall_score: int = Field(ge=0, le=10, description="Overall effectiveness score")
    top_priority_fix: str = Field(description="The single most impactful change to make first")
    improved_prompt: str = Field(description="Complete rewritten prompt that would score 10/10")


_ANSWER_SYSTEM_PROMPT = """You are an expert prompt engineer evaluating a CRM ANSWER system prompt.

The ANSWER prompt's job: Take CRM data and answer user questions accurately.

Score these 4 dimensions (0-10 each):
1. **Grounding**: Does it make hallucination IMPOSSIBLE? Every claim must be traceable to provided data.
2. **Clarity**: Zero ambiguity. No conflicting instructions.
3. **Completeness**: Handles empty data, partial data, stale data, ambiguous questions.
4. **Robustness**: Resists prompt injection, handles malformed input gracefully.

A 10/10 ANSWER prompt must:
- NEVER allow fabricating names, dates, numbers, or relationships not in the data
- Clearly handle "data not available" cases
- Consistent format regardless of data shape
- No conflicting instructions (e.g., "use ALL data" vs "keep concise")

When providing improved_prompt, write the COMPLETE new prompt."""

_ACTION_SYSTEM_PROMPT = """You are an expert prompt engineer evaluating a CRM ACTION system prompt.

The ACTION prompt's job: Suggest next actions based on an answer that was already generated.

Score these 4 dimensions (0-10 each):
1. **Grounding**: Does it ONLY allow referencing entities/facts from the answer? No fabricated names/dates/deadlines.
2. **Actionability**: Are suggested actions useful, concrete, and actually doable by a sales manager?
3. **Specificity**: Does it enforce clear who/what/when WITHOUT fabricating missing details?
4. **Robustness**: Correctly identifies when to say NONE? Handles edge cases?

A 10/10 ACTION prompt must:
- NEVER allow fabricating deadlines, owners, or contacts not in the answer
- Actions must be grounded in the answer content
- Clear criteria for when NONE is appropriate
- Balance specificity with grounding (can't be specific about things not in the data)

When providing improved_prompt, write the COMPLETE new prompt."""

_HUMAN_PROMPT = """## Prompt Type
{prompt_type}

## Current Prompt
```
{current_prompt}
```

## Previous Feedback (if any)
{previous_feedback}

Analyze this prompt and provide:
1. Detailed scores with specific issues and fixes
2. A complete rewritten prompt that would score 10/10

Be specific - point to exact lines/phrases that cause problems."""


def analyze_prompt(prompt_type: str, current_prompt: str, previous_feedback: str = "None", max_retries: int = 3) -> AnswerPromptAnalysis | ActionPromptAnalysis:
    """Get GPT-5.2-pro's detailed analysis of a prompt."""
    import time

    # Use different system prompt and output class based on prompt type
    if prompt_type.lower() == "answer":
        system_prompt = _ANSWER_SYSTEM_PROMPT
        output_class = AnswerPromptAnalysis
    else:
        system_prompt = _ACTION_SYSTEM_PROMPT
        output_class = ActionPromptAnalysis

    chain = create_openai_chain(
        system_prompt=system_prompt,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=8192,  # Increased for longer improved prompts
        structured_output=output_class,
        streaming=False,
        model="gpt-5.2-pro",
        timeout=300,  # 5 minute timeout for GPT-5.2-pro
    )

    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "prompt_type": prompt_type,
                "current_prompt": current_prompt,
                "previous_feedback": previous_feedback,
            })
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(5)
            else:
                raise


def print_analysis(prompt_type: str, analysis: AnswerPromptAnalysis | ActionPromptAnalysis) -> None:
    """Print analysis in a readable format."""

    print(f"\n{'=' * 70}")
    print(f"{prompt_type.upper()} PROMPT ANALYSIS")
    print("=" * 70)

    # Different dimensions for each prompt type
    if prompt_type.lower() == "answer":
        dimensions = [
            ("Grounding", analysis.grounding),
            ("Clarity", analysis.clarity),
            ("Completeness", analysis.completeness),
            ("Robustness", analysis.robustness),
        ]
    else:
        dimensions = [
            ("Grounding", analysis.grounding),
            ("Actionability", analysis.actionability),
            ("Specificity", analysis.specificity),
            ("Robustness", analysis.robustness),
        ]

    total = 0
    for name, score in dimensions:
        total += score.score
        status = "+" if score.score >= 8 else "x"
        print(f"\n{status} {name}: {score.score}/10")
        print(f"  Issue: {score.current_issue}")
        print(f"  Fix: {score.specific_fix}")
        print(f"  After fix: {score.after_fix_score}/10")

    print(f"\n{'-' * 70}")
    print(f"OVERALL: {analysis.overall_score}/10 (sum: {total}/40)")
    print(f"\nTOP PRIORITY: {analysis.top_priority_fix}")

    print(f"\n{'-' * 70}")
    print("IMPROVED PROMPT (10/10):")
    print("-" * 70)
    print(analysis.improved_prompt)


def iterate_to_perfect(prompt_type: str, initial_prompt: str, max_iterations: int = 3) -> tuple[str, list]:
    """Iterate with GPT-5.2-pro until we reach 10/10 or max iterations."""

    current_prompt = initial_prompt
    history: list = []

    for i in range(max_iterations):
        print(f"\n{'#' * 70}")
        print(f"ITERATION {i + 1}/{max_iterations}")
        print("#" * 70)

        # Build feedback from previous iterations
        previous_feedback = "None"
        if history:
            feedback_parts = []
            for j, h in enumerate(history):
                feedback_parts.append(f"Iteration {j + 1}: Overall {h.overall_score}/10")
                feedback_parts.append(f"  Top issue: {h.top_priority_fix}")
            previous_feedback = "\n".join(feedback_parts)

        analysis = analyze_prompt(prompt_type, current_prompt, previous_feedback)
        history.append(analysis)
        print_analysis(prompt_type, analysis)

        # Check if we've reached 10/10
        if analysis.overall_score >= 10:
            print(f"\n✓ ACHIEVED 10/10 after {i + 1} iteration(s)!")
            return analysis.improved_prompt, history

        # Use the improved prompt for next iteration
        current_prompt = analysis.improved_prompt

    print(f"\n⚠ Reached max iterations. Best score: {history[-1].overall_score}/10")
    return current_prompt, history


def save_results(results: dict, output_path: Path) -> None:
    """Save iteration results to JSON."""

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Run prompt improvement iterations."""

    print("=" * 70)
    print("PROMPT IMPROVER - Using GPT-5.2-pro")
    print("Target: 10/10 across all dimensions")
    print("=" * 70)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": "gpt-5.2-pro",
        "prompts": {}
    }

    # Improve Answer prompt
    print("\n" + "=" * 70)
    print("IMPROVING ANSWER PROMPT")
    print("=" * 70)

    final_answer, answer_history = iterate_to_perfect("Answer", ANSWER_PROMPT)
    results["prompts"]["answer"] = {
        "original": ANSWER_PROMPT,
        "final": final_answer,
        "iterations": [
            {
                "overall_score": h.overall_score,
                "grounding": h.grounding.score,
                "clarity": h.clarity.score,
                "completeness": h.completeness.score,
                "robustness": h.robustness.score,
                "top_priority": h.top_priority_fix,
            }
            for h in answer_history
        ]
    }

    # Improve Action prompt
    print("\n" + "=" * 70)
    print("IMPROVING ACTION PROMPT")
    print("=" * 70)

    final_action, action_history = iterate_to_perfect("Action", ACTION_PROMPT)
    results["prompts"]["action"] = {
        "original": ACTION_PROMPT,
        "final": final_action,
        "iterations": [
            {
                "overall_score": h.overall_score,
                "grounding": h.grounding.score,
                "actionability": h.actionability.score,
                "specificity": h.specificity.score,
                "robustness": h.robustness.score,
                "top_priority": h.top_priority_fix,
            }
            for h in action_history
        ]
    }

    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"prompt_improvement_{timestamp}.json"
    save_results(results, output_path)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\nAnswer Prompt:")
    print(f"  Start: {answer_history[0].overall_score}/10 → Final: {answer_history[-1].overall_score}/10")
    print("\nAction Prompt:")
    print(f"  Start: {action_history[0].overall_score}/10 → Final: {action_history[-1].overall_score}/10")

    print("\n" + "=" * 70)
    print("FINAL IMPROVED PROMPTS")
    print("=" * 70)

    print("\n--- ANSWER PROMPT ---")
    print(final_answer)

    print("\n--- ACTION PROMPT ---")
    print(final_action)


if __name__ == "__main__":
    main()
