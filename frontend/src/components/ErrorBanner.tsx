interface ErrorBannerProps {
  message: string;
  onDismiss?: () => void;
}

/**
 * Error banner for displaying API errors
 */
export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="error-banner" role="alert">
      <span>{message}</span>
      {onDismiss && (
        <button
          onClick={onDismiss}
          style={{
            marginLeft: 12,
            background: "none",
            border: "none",
            cursor: "pointer",
            color: "inherit",
            fontSize: "inherit",
          }}
          aria-label="Dismiss error"
        >
          ✕
        </button>
      )}
    </div>
  );
}
