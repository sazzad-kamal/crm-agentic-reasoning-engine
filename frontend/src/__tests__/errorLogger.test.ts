import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  logger,
  configureLogger,
  logErrorBoundary,
  logApiError,
  logUserActionError,
  type LogLevel,
  type ErrorContext,
} from "../utils/errorLogger";

describe("errorLogger", () => {
  // Store original console methods
  const originalConsole = {
    log: console.log,
    warn: console.warn,
    error: console.error,
  };

  // Mock console methods
  beforeEach(() => {
    console.log = vi.fn();
    console.warn = vi.fn();
    console.error = vi.fn();

    // Reset logger config to defaults
    configureLogger({
      minLevel: "debug",
      includeStackTrace: true,
      logToConsole: true,
      customHandler: undefined,
    });
  });

  afterEach(() => {
    console.log = originalConsole.log;
    console.warn = originalConsole.warn;
    console.error = originalConsole.error;
  });

  // ===========================================================================
  // Basic Logging
  // ===========================================================================

  describe("basic logging", () => {
    it("logs debug messages", () => {
      logger.debug("Debug message");
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining("[DEBUG]"),
        ""
      );
    });

    it("logs info messages", () => {
      logger.info("Info message");
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining("[INFO]"),
        ""
      );
    });

    it("logs warning messages", () => {
      logger.warn("Warning message");
      expect(console.warn).toHaveBeenCalledWith(
        expect.stringContaining("Warning message"),
        ""
      );
    });

    it("logs error messages", () => {
      logger.error("Error message");
      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining("Error message"),
        ""
      );
    });

    it("includes timestamp in logs", () => {
      logger.info("Test message");
      expect(console.log).toHaveBeenCalledWith(
        expect.stringMatching(/\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/),
        ""
      );
    });
  });

  // ===========================================================================
  // Log Levels
  // ===========================================================================

  describe("log level filtering", () => {
    it("filters out logs below minLevel", () => {
      configureLogger({ minLevel: "warn" });

      logger.debug("Debug message");
      logger.info("Info message");

      expect(console.log).not.toHaveBeenCalled();
    });

    it("logs messages at or above minLevel", () => {
      configureLogger({ minLevel: "warn" });

      logger.warn("Warning message");
      logger.error("Error message");

      expect(console.warn).toHaveBeenCalled();
      expect(console.error).toHaveBeenCalled();
    });

    it("respects level priority: debug < info < warn < error", () => {
      configureLogger({ minLevel: "info" });

      logger.debug("Debug");
      expect(console.log).not.toHaveBeenCalled();

      logger.info("Info");
      expect(console.log).toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Context
  // ===========================================================================

  describe("error context", () => {
    it("includes component name in log", () => {
      logger.info("Test message", { component: "TestComponent" });
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining("[TestComponent]"),
        ""
      );
    });

    it("includes metadata in log", () => {
      const metadata = { userId: "123", page: "/dashboard" };
      logger.info("Test message", { component: "Test", metadata });
      expect(console.log).toHaveBeenCalledWith(
        expect.any(String),
        metadata
      );
    });

    it("passes full context to custom handler", () => {
      const customHandler = vi.fn();
      configureLogger({ customHandler });

      const context: ErrorContext = {
        component: "MyComponent",
        action: "submit",
        metadata: { formId: "form-1" },
        userId: "user-123",
        sessionId: "session-abc",
      };

      logger.info("Test", context);

      expect(customHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          level: "info",
          message: "Test",
          context,
        })
      );
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe("error handling", () => {
    it("logs Error objects with stack trace", () => {
      const testError = new Error("Test error");
      logger.error("Error occurred", testError);

      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining("Error occurred"),
        ""
      );
      expect(console.error).toHaveBeenCalledWith(testError);
    });

    it("converts string errors to Error objects", () => {
      logger.error("Error occurred", "String error");

      expect(console.error).toHaveBeenCalledWith(
        expect.any(Error)
      );
    });

    it("converts unknown errors to Error objects", () => {
      logger.error("Error occurred", { weird: "object" });

      expect(console.error).toHaveBeenCalledWith(
        expect.any(Error)
      );
    });

    it("captureException extracts message from error", () => {
      const testError = new Error("Captured error message");
      logger.captureException(testError);

      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining("Captured error message"),
        ""
      );
    });
  });

  // ===========================================================================
  // Configuration
  // ===========================================================================

  describe("configuration", () => {
    it("can disable console logging", () => {
      configureLogger({ logToConsole: false });

      logger.info("Test message");
      logger.error("Error message");

      expect(console.log).not.toHaveBeenCalled();
      expect(console.error).not.toHaveBeenCalled();
    });

    it("can add custom handler", () => {
      const customHandler = vi.fn();
      configureLogger({ customHandler });

      logger.info("Test message");

      expect(customHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          level: "info",
          message: "Test message",
        })
      );
    });

    it("handles custom handler errors gracefully", () => {
      const throwingHandler = vi.fn().mockImplementation(() => {
        throw new Error("Handler error");
      });
      configureLogger({ customHandler: throwingHandler });

      // Should not throw
      expect(() => logger.info("Test")).not.toThrow();

      // Should log handler error to console
      expect(console.error).toHaveBeenCalledWith(
        "Error in custom log handler:",
        expect.any(Error)
      );
    });

    it("can disable stack traces", () => {
      configureLogger({ includeStackTrace: false });

      const testError = new Error("Test");
      logger.warn("Warning", testError);

      // Should not log the error object for warnings when stack trace disabled
      expect(console.warn).toHaveBeenCalledTimes(1);
    });
  });

  // ===========================================================================
  // Scoped Logger
  // ===========================================================================

  describe("scoped logger", () => {
    it("creates scoped logger with default context", () => {
      const scopedLogger = logger.scope({ component: "ScopedComponent" });

      scopedLogger.info("Scoped message");

      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining("[ScopedComponent]"),
        ""
      );
    });

    it("scoped logger merges context", () => {
      const customHandler = vi.fn();
      configureLogger({ customHandler });

      const scopedLogger = logger.scope({
        component: "ScopedComponent",
        userId: "default-user",
      });

      scopedLogger.info("Test", { action: "click", userId: "override-user" });

      expect(customHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          context: expect.objectContaining({
            component: "ScopedComponent",
            action: "click",
            userId: "override-user", // Override takes precedence
          }),
        })
      );
    });

    it("scoped logger supports all log methods", () => {
      const scopedLogger = logger.scope({ component: "Test" });

      scopedLogger.debug("Debug");
      scopedLogger.info("Info");
      scopedLogger.warn("Warn");
      scopedLogger.error("Error");
      scopedLogger.captureException(new Error("Exception"));

      expect(console.log).toHaveBeenCalledTimes(2); // debug + info
      expect(console.warn).toHaveBeenCalledTimes(1);
      // error logs twice per call (message + error object) for error and captureException
      expect(console.error).toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Helper Functions
  // ===========================================================================

  describe("logErrorBoundary", () => {
    it("logs error boundary errors", () => {
      const error = new Error("Render error");
      const errorInfo = { componentStack: "\n  at MyComponent" } as React.ErrorInfo;

      logErrorBoundary(error, errorInfo, "MyComponent");

      // Verify console.error was called (it logs message then error object)
      expect(console.error).toHaveBeenCalled();
      // First call includes the message
      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining("React Error Boundary"),
        expect.anything()
      );
    });

    it("includes component name in context", () => {
      const customHandler = vi.fn();
      configureLogger({ customHandler });

      const error = new Error("Test");
      const errorInfo = { componentStack: "" } as React.ErrorInfo;

      logErrorBoundary(error, errorInfo, "CustomBoundary");

      expect(customHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          context: expect.objectContaining({
            component: "CustomBoundary",
          }),
        })
      );
    });

    it("uses default name when not provided", () => {
      const customHandler = vi.fn();
      configureLogger({ customHandler });

      const error = new Error("Test");
      const errorInfo = { componentStack: "" } as React.ErrorInfo;

      logErrorBoundary(error, errorInfo);

      expect(customHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          context: expect.objectContaining({
            component: "ErrorBoundary",
          }),
        })
      );
    });
  });

  describe("logApiError", () => {
    it("logs API errors with status and endpoint", () => {
      const customHandler = vi.fn();
      configureLogger({ customHandler });

      logApiError("/api/users", 500, new Error("Server error"));

      expect(customHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "API error: /api/users",
          context: expect.objectContaining({
            component: "API",
            action: "/api/users",
            metadata: expect.objectContaining({
              status: 500,
              endpoint: "/api/users",
            }),
          }),
        })
      );
    });
  });

  describe("logUserActionError", () => {
    it("logs user action errors with metadata", () => {
      const customHandler = vi.fn();
      configureLogger({ customHandler });

      logUserActionError("submitForm", new Error("Validation failed"), {
        formId: "contact-form",
        fields: ["email"],
      });

      expect(customHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "User action failed: submitForm",
          context: expect.objectContaining({
            component: "UserAction",
            action: "submitForm",
            metadata: expect.objectContaining({
              formId: "contact-form",
              fields: ["email"],
            }),
          }),
        })
      );
    });
  });

  // ===========================================================================
  // Log Entry Structure
  // ===========================================================================

  describe("log entry structure", () => {
    it("creates properly structured log entries", () => {
      const customHandler = vi.fn();
      configureLogger({ customHandler });

      logger.error("Test error", new Error("Details"), {
        component: "Test",
        action: "test-action",
      });

      expect(customHandler).toHaveBeenCalledWith({
        level: "error",
        message: "Test error",
        timestamp: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/),
        error: expect.any(Error),
        context: {
          component: "Test",
          action: "test-action",
        },
      });
    });
  });
});
