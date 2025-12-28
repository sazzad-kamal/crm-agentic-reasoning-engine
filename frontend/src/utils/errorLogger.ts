/**
 * Centralized Error Logging Utility
 *
 * Provides consistent error handling and logging across the application.
 * Supports different log levels, error context, and can be extended to
 * integrate with external error tracking services (e.g., Sentry, LogRocket).
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface ErrorContext {
  /** Component or module where the error occurred */
  component?: string;
  /** Action being performed when the error occurred */
  action?: string;
  /** Additional metadata for debugging */
  metadata?: Record<string, unknown>;
  /** User ID if available */
  userId?: string;
  /** Session ID if available */
  sessionId?: string;
}

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: string;
  error?: Error;
  context?: ErrorContext;
}

/**
 * Configuration for the error logger
 */
interface LoggerConfig {
  /** Minimum log level to output (debug < info < warn < error) */
  minLevel: LogLevel;
  /** Whether to include stack traces in logs */
  includeStackTrace: boolean;
  /** Whether to log to console */
  logToConsole: boolean;
  /** Custom log handler for external services */
  customHandler?: (entry: LogEntry) => void;
}

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

/**
 * Default configuration - can be overridden via configure()
 */
let config: LoggerConfig = {
  minLevel: import.meta.env?.MODE === "production" ? "warn" : "debug",
  includeStackTrace: import.meta.env?.MODE !== "production",
  logToConsole: true,
  customHandler: undefined,
};

/**
 * Configure the error logger
 */
export function configureLogger(newConfig: Partial<LoggerConfig>): void {
  config = { ...config, ...newConfig };
}

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
 * Create a log entry
 */
function createLogEntry(
  level: LogLevel,
  message: string,
  error?: Error,
  context?: ErrorContext
): LogEntry {
  return {
    level,
    message,
    timestamp: new Date().toISOString(),
    error,
    context,
  };
}

/**
 * Output log entry to console
 */
function logToConsole(entry: LogEntry): void {
  const prefix = `[${entry.timestamp}] [${entry.level.toUpperCase()}]`;
  const contextStr = entry.context?.component
    ? ` [${entry.context.component}]`
    : "";

  const fullMessage = `${prefix}${contextStr} ${entry.message}`;

  switch (entry.level) {
    case "debug":
      // eslint-disable-next-line no-console
      console.log(`[DEBUG]${fullMessage}`, entry.context?.metadata || "");
      break;
    case "info":
      // eslint-disable-next-line no-console
      console.log(`[INFO]${fullMessage}`, entry.context?.metadata || "");
      break;
    case "warn":
      console.warn(fullMessage, entry.context?.metadata || "");
      if (entry.error && config.includeStackTrace) {
        console.warn(entry.error);
      }
      break;
    case "error":
      console.error(fullMessage, entry.context?.metadata || "");
      if (entry.error) {
        console.error(entry.error);
      }
      break;
  }
}

/**
 * Process a log entry
 */
function processLog(entry: LogEntry): void {
  // Check if we should log based on level
  if (LOG_LEVEL_PRIORITY[entry.level] < LOG_LEVEL_PRIORITY[config.minLevel]) {
    return;
  }

  // Log to console if enabled
  if (config.logToConsole) {
    logToConsole(entry);
  }

  // Call custom handler if provided (for external services)
  if (config.customHandler) {
    try {
      config.customHandler(entry);
    } catch (handlerError) {
      // Avoid infinite loop - just console.error
      console.error("Error in custom log handler:", handlerError);
    }
  }
}

/**
 * Error Logger API
 */
export const logger = {
  /**
   * Log a debug message
   */
  debug(message: string, context?: ErrorContext): void {
    processLog(createLogEntry("debug", message, undefined, context));
  },

  /**
   * Log an info message
   */
  info(message: string, context?: ErrorContext): void {
    processLog(createLogEntry("info", message, undefined, context));
  },

  /**
   * Log a warning
   */
  warn(message: string, error?: unknown, context?: ErrorContext): void {
    const formattedError = error ? formatError(error) : undefined;
    processLog(createLogEntry("warn", message, formattedError, context));
  },

  /**
   * Log an error
   */
  error(message: string, error?: unknown, context?: ErrorContext): void {
    const formattedError = error ? formatError(error) : undefined;
    processLog(createLogEntry("error", message, formattedError, context));
  },

  /**
   * Log an error with automatic message extraction
   */
  captureException(error: unknown, context?: ErrorContext): void {
    const formattedError = formatError(error);
    processLog(
      createLogEntry("error", formattedError.message, formattedError, context)
    );
  },

  /**
   * Create a scoped logger with preset context
   */
  scope(defaultContext: ErrorContext) {
    return {
      debug: (message: string, context?: ErrorContext) =>
        logger.debug(message, { ...defaultContext, ...context }),
      info: (message: string, context?: ErrorContext) =>
        logger.info(message, { ...defaultContext, ...context }),
      warn: (message: string, error?: unknown, context?: ErrorContext) =>
        logger.warn(message, error, { ...defaultContext, ...context }),
      error: (message: string, error?: unknown, context?: ErrorContext) =>
        logger.error(message, error, { ...defaultContext, ...context }),
      captureException: (error: unknown, context?: ErrorContext) =>
        logger.captureException(error, { ...defaultContext, ...context }),
    };
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

/**
 * API error helper
 */
export function logApiError(
  endpoint: string,
  status: number,
  error: unknown
): void {
  logger.error(`API error: ${endpoint}`, error, {
    component: "API",
    action: endpoint,
    metadata: {
      status,
      endpoint,
    },
  });
}

/**
 * User action error helper
 */
export function logUserActionError(
  action: string,
  error: unknown,
  metadata?: Record<string, unknown>
): void {
  logger.error(`User action failed: ${action}`, error, {
    component: "UserAction",
    action,
    metadata,
  });
}

export default logger;
