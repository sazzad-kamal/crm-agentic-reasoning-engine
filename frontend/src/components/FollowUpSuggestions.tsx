import type { KeyboardEvent } from "react";

interface FollowUpSuggestionsProps {
  suggestions: string[];
  onSuggestionClick: (suggestion: string) => void;
}

/**
 * Displays follow-up question suggestions as clickable chips
 */
export function FollowUpSuggestions({
  suggestions,
  onSuggestionClick,
}: FollowUpSuggestionsProps) {
  if (!suggestions || suggestions.length === 0) return null;

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, suggestion: string) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSuggestionClick(suggestion);
    }
  };

  return (
    <div
      className="flex-row follow-up-container"
      role="group"
      aria-label="Suggested follow-up questions"
    >
      <span className="follow-up-container__label" aria-hidden="true">
        Ask:
      </span>
      {suggestions.map((suggestion, idx) => (
        <button
          key={idx}
          className="follow-up-chip"
          onClick={() => onSuggestionClick(suggestion)}
          onKeyDown={(e) => handleKeyDown(e, suggestion)}
          type="button"
          aria-label={`Ask follow-up: ${suggestion}`}
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}
