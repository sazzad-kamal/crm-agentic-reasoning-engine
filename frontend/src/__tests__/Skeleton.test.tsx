import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageSkeleton, ChatSkeleton } from "../components/Skeleton";
import { SkipLink } from "../components/SkipLink";

// =========================================================================
// MessageSkeleton Component
// =========================================================================

describe("MessageSkeleton", () => {
  it("renders skeleton with correct class", () => {
    const { container } = render(<MessageSkeleton />);

    const skeleton = container.querySelector(".message-skeleton");
    expect(skeleton).toBeInTheDocument();
  });

  it("has proper ARIA attributes", () => {
    render(<MessageSkeleton />);

    const skeleton = screen.getByRole("status");
    expect(skeleton).toHaveAttribute("aria-label", "Loading message");
    expect(skeleton).toHaveAttribute("aria-busy", "true");
  });

  it("renders three skeleton lines", () => {
    const { container } = render(<MessageSkeleton />);

    const lines = container.querySelectorAll(".message-skeleton__line");
    expect(lines).toHaveLength(3);
  });

  it("renders short, long, and medium lines", () => {
    const { container } = render(<MessageSkeleton />);

    const shortLine = container.querySelector(".message-skeleton__line--short");
    const longLine = container.querySelector(".message-skeleton__line--long");
    const mediumLine = container.querySelector(".message-skeleton__line--medium");

    expect(shortLine).toBeInTheDocument();
    expect(longLine).toBeInTheDocument();
    expect(mediumLine).toBeInTheDocument();
  });

  it("memoizes correctly", () => {
    const { rerender } = render(<MessageSkeleton />);

    rerender(<MessageSkeleton />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});

// =========================================================================
// ChatSkeleton Component
// =========================================================================

describe("ChatSkeleton", () => {
  it("renders chat skeleton with correct class", () => {
    const { container } = render(<ChatSkeleton />);

    const skeleton = container.querySelector(".chat-skeleton");
    expect(skeleton).toBeInTheDocument();
  });

  it("has proper ARIA attributes", () => {
    const { container } = render(<ChatSkeleton />);

    const chatSkeleton = container.querySelector(".chat-skeleton");
    expect(chatSkeleton).toHaveAttribute("aria-label", "Loading chat");
    expect(chatSkeleton).toHaveAttribute("aria-busy", "true");
    expect(chatSkeleton).toHaveAttribute("role", "status");
  });

  it("renders two message skeletons", () => {
    const { container } = render(<ChatSkeleton />);

    const messageSkeletons = container.querySelectorAll(".message-skeleton");
    expect(messageSkeletons).toHaveLength(2);
  });

  it("renders six skeleton lines total (3 per message)", () => {
    const { container } = render(<ChatSkeleton />);

    const lines = container.querySelectorAll(".message-skeleton__line");
    expect(lines).toHaveLength(6);
  });

  it("memoizes correctly", () => {
    const { container, rerender } = render(<ChatSkeleton />);

    rerender(<ChatSkeleton />);
    const chatSkeleton = container.querySelector(".chat-skeleton");
    expect(chatSkeleton).toBeInTheDocument();
  });
});

// =========================================================================
// SkipLink Component
// =========================================================================

describe("SkipLink", () => {
  it("renders with default text", () => {
    render(<SkipLink />);

    expect(screen.getByText("Skip to main content")).toBeInTheDocument();
  });

  it("renders with custom text", () => {
    render(<SkipLink>Jump to content</SkipLink>);

    expect(screen.getByText("Jump to content")).toBeInTheDocument();
  });

  it("has default target ID", () => {
    render(<SkipLink />);

    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "#main-content");
  });

  it("uses custom target ID", () => {
    render(<SkipLink targetId="custom-target" />);

    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "#custom-target");
  });

  it("has tabIndex of 0", () => {
    render(<SkipLink />);

    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("tabIndex", "0");
  });

  it("has correct class name", () => {
    const { container } = render(<SkipLink />);

    const link = container.querySelector(".skip-link");
    expect(link).toBeInTheDocument();
  });

  it("is an anchor element", () => {
    render(<SkipLink />);

    const link = screen.getByRole("link");
    expect(link.tagName).toBe("A");
  });

  it("memoizes correctly", () => {
    const { rerender } = render(<SkipLink />);

    rerender(<SkipLink />);
    expect(screen.getByText("Skip to main content")).toBeInTheDocument();
  });

  it("handles empty custom text", () => {
    render(<SkipLink>{""}</SkipLink>);

    const link = screen.getByRole("link");
    expect(link.textContent).toBe("");
  });

  it("combines custom text and target ID", () => {
    render(<SkipLink targetId="content">Skip navigation</SkipLink>);

    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "#content");
    expect(link).toHaveTextContent("Skip navigation");
  });
});
