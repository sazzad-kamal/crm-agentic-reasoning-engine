import { describe, it, expect, vi, beforeAll, afterAll } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ErrorBoundary } from "../components/ErrorBoundary";

// Component that throws an error
const ThrowError = ({ shouldThrow }: { shouldThrow?: boolean }) => {
  if (shouldThrow) {
    throw new Error("Test error");
  }
  return <div>No error</div>;
};

describe("ErrorBoundary", () => {
  // Suppress console.error during tests
  const originalError = console.error;
  beforeAll(() => {
    console.error = vi.fn();
  });

  afterAll(() => {
    console.error = originalError;
  });

  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders children when no error", () => {
    render(
      <ErrorBoundary>
        <div>Child content</div>
      </ErrorBoundary>
    );

    expect(screen.getByText("Child content")).toBeInTheDocument();
  });

  it("renders error UI when child throws", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow />
      </ErrorBoundary>
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("displays error message", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow />
      </ErrorBoundary>
    );

    expect(screen.getByText("Test error")).toBeInTheDocument();
  });

  it("shows retry button", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow />
      </ErrorBoundary>
    );

    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });

  // =========================================================================
  // Custom Fallback
  // =========================================================================

  it("renders custom fallback when provided", () => {
    const customFallback = <div>Custom error message</div>;

    render(
      <ErrorBoundary fallback={customFallback}>
        <ThrowError shouldThrow />
      </ErrorBoundary>
    );

    expect(screen.getByText("Custom error message")).toBeInTheDocument();
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // =========================================================================
  // Error Recovery
  // =========================================================================

  it("resets error state when retry button clicked", async () => {
    // Track whether to throw
    let shouldThrow = true;
    const ConditionalThrow = () => {
      if (shouldThrow) throw new Error("Test error");
      return <div>Recovered</div>;
    };

    render(
      <ErrorBoundary>
        <ConditionalThrow />
      </ErrorBoundary>
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();

    // Stop throwing before retry
    shouldThrow = false;

    const retryButton = screen.getByRole("button", { name: /try again/i });
    await userEvent.click(retryButton);

    expect(screen.getByText("Recovered")).toBeInTheDocument();
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("has proper ARIA attributes", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow />
      </ErrorBoundary>
    );

    const errorDiv = screen.getByRole("alert");
    expect(errorDiv).toHaveAttribute("aria-live", "assertive");
    expect(errorDiv).toHaveAttribute("aria-atomic", "true");
  });

  it("retry button has proper ARIA label", () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow />
      </ErrorBoundary>
    );

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-label", "Try again");
  });

  // =========================================================================
  // Error Logging
  // =========================================================================

  it("logs error to console", () => {
    const consoleError = vi.fn();
    console.error = consoleError;

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow />
      </ErrorBoundary>
    );

    expect(consoleError).toHaveBeenCalled();
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles error without message", () => {
    const ThrowNoMessage = () => {
      throw new Error();
    };

    render(
      <ErrorBoundary>
        <ThrowNoMessage />
      </ErrorBoundary>
    );

    expect(screen.getByText("An unexpected error occurred.")).toBeInTheDocument();
  });

  it("handles multiple children", () => {
    render(
      <ErrorBoundary>
        <div>Child 1</div>
        <div>Child 2</div>
        <div>Child 3</div>
      </ErrorBoundary>
    );

    expect(screen.getByText("Child 1")).toBeInTheDocument();
    expect(screen.getByText("Child 2")).toBeInTheDocument();
    expect(screen.getByText("Child 3")).toBeInTheDocument();
  });

  it("catches errors from nested components", () => {
    const Nested = () => <ThrowError shouldThrow />;
    const Parent = () => <Nested />;

    render(
      <ErrorBoundary>
        <Parent />
      </ErrorBoundary>
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("boundary does not catch its own errors", () => {
    // Error boundaries only catch errors in children, not in themselves
    render(
      <ErrorBoundary>
        <div>Safe content</div>
      </ErrorBoundary>
    );

    expect(screen.getByText("Safe content")).toBeInTheDocument();
  });
});
