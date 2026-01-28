interface SuggestedActionsProps {
  action: string;
}

/**
 * Displays a suggested action as a highlighted callout
 */
export function SuggestedActions({ action }: SuggestedActionsProps) {
  return (
    <div
      className="suggested-actions"
      role="complementary"
      aria-label="Suggested action"
    >
      <span className="suggested-actions__label">Suggested action:</span>
      <span className="suggested-actions__item">{action}</span>
    </div>
  );
}
