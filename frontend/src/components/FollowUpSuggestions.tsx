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

  return (
    <div className="follow-up-container">
      <span className="follow-up-container__label">💡 Follow-up questions:</span>
      {suggestions.map((suggestion, idx) => (
        <button
          key={idx}
          className="follow-up-chip"
          onClick={() => onSuggestionClick(suggestion)}
          type="button"
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}
