/* eslint-disable no-console */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { logger, logErrorBoundary } from "../utils/errorLogger";

describe("errorLogger", () => {
  const originalConsole = {
    log: console.log,
    warn: console.warn,
    error: console.error,
  };

  beforeEach(() => {
    console.log = vi.fn();
    console.warn = vi.fn();
    console.error = vi.fn();
  });

  afterEach(() => {
    console.log = originalConsole.log;
    console.warn = originalConsole.warn;
    console.error = originalConsole.error;
  });

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
      expect(console.log).toHaveBeenCalledWith(expect.any(String), metadata);
    });
  });

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
      expect(console.error).toHaveBeenCalledWith(expect.any(Error));
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

  describe("logErrorBoundary", () => {
    it("logs error boundary errors", () => {
      const error = new Error("Render error");
      const errorInfo = {
        componentStack: "\n  at MyComponent",
      } as React.ErrorInfo;

      logErrorBoundary(error, errorInfo, "MyComponent");

      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining("React Error Boundary"),
        expect.anything()
      );
    });

    it("includes component name in context", () => {
      const error = new Error("Test");
      const errorInfo = { componentStack: "" } as React.ErrorInfo;

      logErrorBoundary(error, errorInfo, "CustomBoundary");

      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining("[CustomBoundary]"),
        expect.anything()
      );
    });

    it("uses default name when not provided", () => {
      const error = new Error("Test");
      const errorInfo = { componentStack: "" } as React.ErrorInfo;

      logErrorBoundary(error, errorInfo);

      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining("[ErrorBoundary]"),
        expect.anything()
      );
    });
  });
});
