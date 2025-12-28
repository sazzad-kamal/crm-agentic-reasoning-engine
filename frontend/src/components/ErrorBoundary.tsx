import { Component, ReactNode } from "react";
import { logErrorBoundary } from "../utils/errorLogger";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  /** Optional name for this boundary for better error tracking */
  name?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorId: string | null;
}

/**
 * Error Boundary component to catch and handle React rendering errors.
 * Prevents the entire app from crashing when a component fails.
 * Integrates with centralized error logging for consistent tracking.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null, errorId: null };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Generate a unique error ID for tracking
    const errorId = `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return { hasError: true, error, errorId };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // Use centralized error logging
    logErrorBoundary(error, errorInfo, this.props.name);
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null, errorId: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          className="error-boundary"
          role="alert"
          aria-live="assertive"
          aria-atomic="true"
        >
          <div className="error-boundary__content">
            <h2 className="error-boundary__title">Something went wrong</h2>
            <p className="error-boundary__message">
              {this.state.error?.message || "An unexpected error occurred."}
            </p>
            <button
              className="btn btn--primary error-boundary__button"
              onClick={this.handleRetry}
              type="button"
              aria-label="Try again"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
