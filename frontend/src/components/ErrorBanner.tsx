interface ErrorBannerProps {
  message: string;
  onDismiss?: () => void;
}

/**
 * Error banner for displaying API errors
 */
export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="error-banner" role="alert" aria-live="assertive">
      <span className="error-banner__message">{message}</span>
      {onDismiss && (
        <button
          className="error-banner__dismiss"
          onClick={onDismiss}
          type="button"
          aria-label="Dismiss error"
        >
          ✕
        </button>
      )}
    </div>
  );
}
