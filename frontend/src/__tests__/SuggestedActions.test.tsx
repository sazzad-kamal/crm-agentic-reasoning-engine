import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SuggestedActions } from "../components/SuggestedActions";

describe("SuggestedActions", () => {
  it("renders the action text", () => {
    render(<SuggestedActions action="Schedule a follow-up call" />);

    expect(screen.getByText("Schedule a follow-up call")).toBeInTheDocument();
  });

  it("renders the label", () => {
    render(<SuggestedActions action="Review pipeline" />);

    expect(screen.getByText("Suggested actions:")).toBeInTheDocument();
  });

  it("has complementary role", () => {
    render(<SuggestedActions action="Test action" />);

    expect(screen.getByRole("complementary", { name: /suggested action/i })).toBeInTheDocument();
  });

  it("has proper ARIA label", () => {
    render(<SuggestedActions action="Test action" />);

    const element = screen.getByRole("complementary");
    expect(element).toHaveAttribute("aria-label", "Suggested actions");
  });

  it("renders with correct class names", () => {
    const { container } = render(<SuggestedActions action="Test" />);

    expect(container.querySelector(".suggested-actions")).toBeInTheDocument();
    expect(container.querySelector(".suggested-actions__label")).toBeInTheDocument();
    expect(container.querySelector(".suggested-actions__content")).toBeInTheDocument();
  });
});
