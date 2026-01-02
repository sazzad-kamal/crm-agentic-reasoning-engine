/**
 * Centralized Error Logging Utility
 *
 * Provides consistent error handling and logging across the application.
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface ErrorContext {
  /** Component or module where the error occurred */
  component?: string;
  /** Action being performed when the error occurred */
  action?: string;
  /** Additional metadata for debugging */
  metadata?: Record<string, unknown>;
}

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const minLevel: LogLevel =
  import.meta.env?.MODE === "production" ? "warn" : "debug";
const includeStackTrace = import.meta.env?.MODE !== "production";

/**
 * Format error for logging
 */
function formatError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }
  if (typeof error === "string") {
    return new Error(error);
  }
  return new Error(String(error));
}

/**
 * Output log to console
 */
function logToConsole(
  level: LogLevel,
  message: string,
  error?: Error,
  context?: ErrorContext
): void {
  if (LOG_LEVEL_PRIORITY[level] < LOG_LEVEL_PRIORITY[minLevel]) {
    return;
  }

  const prefix = `[${new Date().toISOString()}] [${level.toUpperCase()}]`;
  const contextStr = context?.component ? ` [${context.component}]` : "";
  const fullMessage = `${prefix}${contextStr} ${message}`;

  switch (level) {
    case "debug":
    case "info":
      // eslint-disable-next-line no-console
      console.log(fullMessage, context?.metadata || "");
      break;
    case "warn":
      console.warn(fullMessage, context?.metadata || "");
      if (error && includeStackTrace) {
        console.warn(error);
      }
      break;
    case "error":
      console.error(fullMessage, context?.metadata || "");
      if (error) {
        console.error(error);
      }
      break;
  }
}

/**
 * Error Logger API
 */
export const logger = {
  debug(message: string, context?: ErrorContext): void {
    logToConsole("debug", message, undefined, context);
  },

  info(message: string, context?: ErrorContext): void {
    logToConsole("info", message, undefined, context);
  },

  warn(message: string, error?: unknown, context?: ErrorContext): void {
    logToConsole("warn", message, error ? formatError(error) : undefined, context);
  },

  error(message: string, error?: unknown, context?: ErrorContext): void {
    logToConsole("error", message, error ? formatError(error) : undefined, context);
  },

  captureException(error: unknown, context?: ErrorContext): void {
    const formattedError = formatError(error);
    logToConsole("error", formattedError.message, formattedError, context);
  },
};

/**
 * React Error Boundary helper - call this in componentDidCatch
 */
export function logErrorBoundary(
  error: Error,
  errorInfo: React.ErrorInfo,
  componentName?: string
): void {
  logger.error("React Error Boundary caught error", error, {
    component: componentName || "ErrorBoundary",
    action: "render",
    metadata: {
      componentStack: errorInfo.componentStack,
    },
  });
}

export default logger;
