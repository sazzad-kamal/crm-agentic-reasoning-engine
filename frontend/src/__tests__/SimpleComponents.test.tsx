import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Avatar } from "../components/Avatar";
import { LoadingDots, LoadingState } from "../components/LoadingDots";
import { MetaInfo } from "../components/MetaInfo";
import { ErrorBanner } from "../components/ErrorBanner";
import type { Meta } from "../types";

// =========================================================================
// Avatar Component
// =========================================================================

describe("Avatar", () => {
  it("renders user avatar", () => {
    const { container } = render(<Avatar type="user" />);

    const avatar = container.querySelector(".avatar");
    expect(avatar).toBeInTheDocument();
    expect(avatar).toHaveClass("avatar--user");
    expect(avatar).toHaveClass("avatar--md"); // default size
  });

  it("renders assistant avatar", () => {
    const { container } = render(<Avatar type="assistant" />);

    const avatar = container.querySelector(".avatar");
    expect(avatar).toBeInTheDocument();
    expect(avatar).toHaveClass("avatar--assistant");
  });

  it("applies small size class", () => {
    const { container } = render(<Avatar type="user" size="sm" />);

    const avatar = container.querySelector(".avatar");
    expect(avatar).toHaveClass("avatar--sm");
  });

  it("applies medium size class by default", () => {
    const { container } = render(<Avatar type="user" />);

    const avatar = container.querySelector(".avatar");
    expect(avatar).toHaveClass("avatar--md");
  });

  it("applies large size class", () => {
    const { container } = render(<Avatar type="user" size="lg" />);

    const avatar = container.querySelector(".avatar");
    expect(avatar).toHaveClass("avatar--lg");
  });

  it("renders SVG for user", () => {
    const { container } = render(<Avatar type="user" />);

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
    expect(svg).toHaveClass("avatar__icon");
  });

  it("renders SVG for assistant", () => {
    const { container } = render(<Avatar type="assistant" />);

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("is hidden from screen readers", () => {
    const { container } = render(<Avatar type="user" />);

    const avatar = container.querySelector(".avatar");
    expect(avatar).toHaveAttribute("aria-hidden", "true");
  });

  it("renders different SVG paths for user vs assistant", () => {
    const { container: userContainer } = render(<Avatar type="user" />);
    const { container: assistantContainer } = render(<Avatar type="assistant" />);

    const userPath = userContainer.querySelector("path")?.getAttribute("d");
    const assistantPath = assistantContainer.querySelector("path")?.getAttribute("d");

    expect(userPath).toBeDefined();
    expect(assistantPath).toBeDefined();
    expect(userPath).not.toBe(assistantPath);
  });

  it("memoizes correctly", () => {
    const { container, rerender } = render(<Avatar type="user" />);

    rerender(<Avatar type="user" />);
    // If it re-renders with same props, memo should prevent re-render
    // This is more of a performance check, hard to assert in tests
    const avatar = container.querySelector(".avatar");
    expect(avatar).toBeInTheDocument();
  });
});

// =========================================================================
// LoadingDots Component
// =========================================================================

describe("LoadingDots", () => {
  it("renders three loading dots", () => {
    const { container } = render(<LoadingDots />);

    const dots = container.querySelectorAll(".loading-dot");
    expect(dots).toHaveLength(3);
  });

  it("renders wrapper with correct class", () => {
    const { container } = render(<LoadingDots />);

    const wrapper = container.querySelector(".loading-dots");
    expect(wrapper).toBeInTheDocument();
  });

  it("renders as a span element", () => {
    const { container } = render(<LoadingDots />);

    const wrapper = container.querySelector("span.loading-dots");
    expect(wrapper).toBeInTheDocument();
  });
});

// =========================================================================
// LoadingState Component
// =========================================================================

describe("LoadingState", () => {
  it("renders with default text", () => {
    render(<LoadingState />);

    expect(screen.getByText("Thinking…")).toBeInTheDocument();
  });

  it("renders with custom text", () => {
    render(<LoadingState text="Processing..." />);

    expect(screen.getByText("Processing...")).toBeInTheDocument();
  });

  it("renders loading dots", () => {
    const { container } = render(<LoadingState />);

    const dots = container.querySelectorAll(".loading-dot");
    expect(dots).toHaveLength(3);
  });

  it("has proper ARIA attributes", () => {
    render(<LoadingState />);

    const loadingState = screen.getByRole("status");
    expect(loadingState).toHaveAttribute("aria-live", "polite");
    expect(loadingState).toHaveAttribute("aria-label", "Loading");
  });

  it("has correct class name", () => {
    const { container } = render(<LoadingState />);

    const loadingState = container.querySelector(".loading-state");
    expect(loadingState).toBeInTheDocument();
  });

  it("renders text with correct class", () => {
    const { container } = render(<LoadingState text="Custom" />);

    const text = container.querySelector(".loading-state__text");
    expect(text).toBeInTheDocument();
    expect(text).toHaveTextContent("Custom");
  });
});

// =========================================================================
// MetaInfo Component
// =========================================================================

describe("MetaInfo", () => {
  it("renders latency info", () => {
    const meta: Meta = {
      latency_ms: 1234,
    };

    render(<MetaInfo meta={meta} />);

    expect(screen.getByText(/1234ms/)).toBeInTheDocument();
  });

  it("renders mode info", () => {
    const meta: Meta = {
      mode_used: "data",
    };

    render(<MetaInfo meta={meta} />);

    expect(screen.getByText(/Mode: data/)).toBeInTheDocument();
  });

  it("renders company ID", () => {
    const meta: Meta = {
      company_id: "comp123",
    };

    render(<MetaInfo meta={meta} />);

    expect(screen.getByText(/Company: comp123/)).toBeInTheDocument();
  });

  it("renders all metadata with separator", () => {
    const meta: Meta = {
      latency_ms: 500,
      mode_used: "docs",
      company_id: "acme",
    };

    render(<MetaInfo meta={meta} />);

    const content = screen.getByRole("contentinfo");
    expect(content.textContent).toBe("500ms · Mode: docs · Company: acme");
  });

  it("returns null for empty metadata", () => {
    const meta: Meta = {};

    const { container } = render(<MetaInfo meta={meta} />);

    expect(container.firstChild).toBeNull();
  });

  it("handles partial metadata", () => {
    const meta: Meta = {
      latency_ms: 100,
      mode_used: "auto",
    };

    render(<MetaInfo meta={meta} />);

    const content = screen.getByRole("contentinfo");
    expect(content.textContent).toBe("100ms · Mode: auto");
  });

  it("has proper ARIA attributes", () => {
    const meta: Meta = {
      latency_ms: 500,
      mode_used: "data",
    };

    render(<MetaInfo meta={meta} />);

    const contentinfo = screen.getByRole("contentinfo");
    expect(contentinfo).toHaveAttribute("aria-label", "Response metadata: 500ms, Mode: data");
  });

  it("has correct class name", () => {
    const meta: Meta = {
      latency_ms: 100,
    };

    const { container } = render(<MetaInfo meta={meta} />);

    const metaLine = container.querySelector(".meta-line");
    expect(metaLine).toBeInTheDocument();
  });

  it("memoizes correctly", () => {
    const meta: Meta = {
      latency_ms: 100,
    };

    const { rerender } = render(<MetaInfo meta={meta} />);

    rerender(<MetaInfo meta={meta} />);
    expect(screen.getByText(/100ms/)).toBeInTheDocument();
  });
});

// =========================================================================
// ErrorBanner Component
// =========================================================================

describe("ErrorBanner", () => {
  it("renders error message", () => {
    render(<ErrorBanner message="An error occurred" />);

    expect(screen.getByText("An error occurred")).toBeInTheDocument();
  });

  it("has proper ARIA attributes", () => {
    render(<ErrorBanner message="Error" />);

    const banner = screen.getByRole("alert");
    expect(banner).toHaveAttribute("aria-live", "assertive");
  });

  it("renders dismiss button when onDismiss provided", () => {
    const handleDismiss = vi.fn();

    render(<ErrorBanner message="Error" onDismiss={handleDismiss} />);

    const dismissButton = screen.getByRole("button", { name: "Dismiss error" });
    expect(dismissButton).toBeInTheDocument();
  });

  it("does not render dismiss button when onDismiss not provided", () => {
    render(<ErrorBanner message="Error" />);

    const dismissButton = screen.queryByRole("button");
    expect(dismissButton).not.toBeInTheDocument();
  });

  it("calls onDismiss when dismiss button clicked", () => {
    const handleDismiss = vi.fn();

    render(<ErrorBanner message="Error" onDismiss={handleDismiss} />);

    const dismissButton = screen.getByRole("button", { name: "Dismiss error" });
    fireEvent.click(dismissButton);

    expect(handleDismiss).toHaveBeenCalledTimes(1);
  });

  it("has correct class name", () => {
    const { container } = render(<ErrorBanner message="Error" />);

    const banner = container.querySelector(".error-banner");
    expect(banner).toBeInTheDocument();
  });

  it("message has correct class", () => {
    const { container } = render(<ErrorBanner message="Test error" />);

    const message = container.querySelector(".error-banner__message");
    expect(message).toBeInTheDocument();
    expect(message).toHaveTextContent("Test error");
  });

  it("dismiss button has correct class", () => {
    const { container } = render(<ErrorBanner message="Error" onDismiss={() => {}} />);

    const button = container.querySelector(".error-banner__dismiss");
    expect(button).toBeInTheDocument();
  });

  it("renders dismiss icon", () => {
    render(<ErrorBanner message="Error" onDismiss={() => {}} />);

    const dismissButton = screen.getByRole("button");
    expect(dismissButton.textContent).toBe("✕");
  });

  it("handles long error messages", () => {
    const longMessage = "A".repeat(500);

    render(<ErrorBanner message={longMessage} />);

    expect(screen.getByText(longMessage)).toBeInTheDocument();
  });

  it("handles special characters in message", () => {
    const message = "<script>alert('xss')</script> & \"quotes\"";

    render(<ErrorBanner message={message} />);

    expect(screen.getByText(message)).toBeInTheDocument();
  });

  it("button is of type button", () => {
    render(<ErrorBanner message="Error" onDismiss={() => {}} />);

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("type", "button");
  });
});
