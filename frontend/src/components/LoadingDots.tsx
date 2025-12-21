/**
 * Loading dots animation component
 */
export function LoadingDots() {
  return (
    <span className="loading-dots">
      <span className="loading-dot" />
      <span className="loading-dot" />
      <span className="loading-dot" />
    </span>
  );
}

/**
 * Loading state with text
 */
export function LoadingState({ text = "Thinking…" }: { text?: string }) {
  return (
    <div
      className="loading-state"
      role="status"
      aria-live="polite"
      aria-label="Loading"
    >
      <LoadingDots />
      <span className="loading-state__text">{text}</span>
    </div>
  );
}
