import { MarkdownText } from "./MarkdownText";

interface SuggestedActionsProps {
  action: string;
}

/**
 * Displays a suggested action plan as a highlighted callout with markdown rendering
 */
export function SuggestedActions({ action }: SuggestedActionsProps) {
  return (
    <div
      className="suggested-actions"
      role="complementary"
      aria-label="Suggested actions"
    >
      <span className="suggested-actions__label">Suggested actions:</span>
      <MarkdownText text={action} className="suggested-actions__content" />
    </div>
  );
}
