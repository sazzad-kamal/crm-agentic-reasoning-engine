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
    <div className="loading-state">
      <LoadingDots />
      <span className="loading-state__text">{text}</span>
    </div>
  );
}
