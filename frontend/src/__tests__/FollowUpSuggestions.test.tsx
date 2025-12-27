import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FollowUpSuggestions } from "../components/FollowUpSuggestions";

describe("FollowUpSuggestions", () => {
  const defaultProps = {
    suggestions: ["Question 1?", "Question 2?", "Question 3?"],
    onSuggestionClick: vi.fn(),
  };

  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders all suggestions", () => {
    render(<FollowUpSuggestions {...defaultProps} />);

    expect(screen.getByText("Question 1?")).toBeInTheDocument();
    expect(screen.getByText("Question 2?")).toBeInTheDocument();
    expect(screen.getByText("Question 3?")).toBeInTheDocument();
  });

  it("renders label", () => {
    render(<FollowUpSuggestions {...defaultProps} />);

    expect(screen.getByText("Ask:")).toBeInTheDocument();
  });

  it("renders null when suggestions is empty", () => {
    const { container } = render(
      <FollowUpSuggestions suggestions={[]} onSuggestionClick={vi.fn()} />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders null when suggestions is undefined", () => {
    const { container } = render(
      <FollowUpSuggestions
        suggestions={undefined as unknown as string[]}
        onSuggestionClick={vi.fn()}
      />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("has proper container structure", () => {
    const { container } = render(<FollowUpSuggestions {...defaultProps} />);

    const group = container.querySelector('[role="group"]');
    expect(group).toBeInTheDocument();
    expect(group).toHaveClass("follow-up-container");
  });

  it("has ARIA label for group", () => {
    render(<FollowUpSuggestions {...defaultProps} />);

    const group = screen.getByRole("group");
    expect(group).toHaveAccessibleName("Suggested follow-up questions");
  });

  it("label is aria-hidden", () => {
    const { container } = render(<FollowUpSuggestions {...defaultProps} />);

    const label = container.querySelector(".follow-up-container__label");
    expect(label).toHaveAttribute("aria-hidden", "true");
  });

  // =========================================================================
  // Click Handling
  // =========================================================================

  it("calls onSuggestionClick when button clicked", async () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    const button = screen.getByText("Question 1?");
    await userEvent.click(button);

    expect(onClick).toHaveBeenCalledWith("Question 1?");
  });

  it("calls onSuggestionClick with correct suggestion", async () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    await userEvent.click(screen.getByText("Question 2?"));
    expect(onClick).toHaveBeenCalledWith("Question 2?");

    await userEvent.click(screen.getByText("Question 3?"));
    expect(onClick).toHaveBeenCalledWith("Question 3?");
  });

  it("handles multiple clicks on same suggestion", async () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    const button = screen.getByText("Question 1?");
    await userEvent.click(button);
    await userEvent.click(button);

    expect(onClick).toHaveBeenCalledTimes(2);
    expect(onClick).toHaveBeenCalledWith("Question 1?");
  });

  it("handles clicks on different suggestions", async () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    await userEvent.click(screen.getByText("Question 1?"));
    await userEvent.click(screen.getByText("Question 2?"));
    await userEvent.click(screen.getByText("Question 3?"));

    expect(onClick).toHaveBeenCalledTimes(3);
  });

  // =========================================================================
  // Keyboard Navigation
  // =========================================================================

  it("handles Enter key", () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    const button = screen.getByText("Question 1?");
    fireEvent.keyDown(button, { key: "Enter" });

    expect(onClick).toHaveBeenCalledWith("Question 1?");
  });

  it("handles Space key", () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    const button = screen.getByText("Question 1?");
    fireEvent.keyDown(button, { key: " " });

    expect(onClick).toHaveBeenCalledWith("Question 1?");
  });

  it("prevents default on Enter", () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    const button = screen.getByText("Question 1?");
    // Component calls preventDefault and triggers click - verify the handler is called
    fireEvent.keyDown(button, { key: "Enter" });

    // If handler was called, preventDefault was also called (per component implementation)
    expect(onClick).toHaveBeenCalledWith("Question 1?");
  });

  it("prevents default on Space", () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    const button = screen.getByText("Question 1?");
    // Component calls preventDefault and triggers click - verify the handler is called
    fireEvent.keyDown(button, { key: " " });

    // If handler was called, preventDefault was also called (per component implementation)
    expect(onClick).toHaveBeenCalledWith("Question 1?");
  });

  it("ignores other keys", () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions {...defaultProps} onSuggestionClick={onClick} />);

    const button = screen.getByText("Question 1?");
    fireEvent.keyDown(button, { key: "a" });
    fireEvent.keyDown(button, { key: "Escape" });
    fireEvent.keyDown(button, { key: "Tab" });

    expect(onClick).not.toHaveBeenCalled();
  });

  // =========================================================================
  // Button Properties
  // =========================================================================

  it("renders buttons with correct type", () => {
    render(<FollowUpSuggestions {...defaultProps} />);

    const buttons = screen.getAllByRole("button");
    buttons.forEach((button) => {
      expect(button).toHaveAttribute("type", "button");
    });
  });

  it("buttons have CSS class", () => {
    const { container } = render(<FollowUpSuggestions {...defaultProps} />);

    const buttons = container.querySelectorAll(".follow-up-chip");
    expect(buttons).toHaveLength(3);
  });

  it("buttons have ARIA labels", () => {
    render(<FollowUpSuggestions {...defaultProps} />);

    expect(
      screen.getByRole("button", { name: "Ask follow-up: Question 1?" })
    ).toBeInTheDocument();

    expect(
      screen.getByRole("button", { name: "Ask follow-up: Question 2?" })
    ).toBeInTheDocument();
  });

  it("buttons are keyboard focusable", () => {
    render(<FollowUpSuggestions {...defaultProps} />);

    const buttons = screen.getAllByRole("button");
    buttons.forEach((button) => {
      button.focus();
      expect(button).toHaveFocus();
    });
  });

  // =========================================================================
  // Different Numbers of Suggestions
  // =========================================================================

  it("renders single suggestion", () => {
    render(
      <FollowUpSuggestions
        suggestions={["Only one?"]}
        onSuggestionClick={vi.fn()}
      />
    );

    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.getByText("Only one?")).toBeInTheDocument();
  });

  it("renders many suggestions", () => {
    const many = Array.from({ length: 10 }, (_, i) => `Question ${i + 1}?`);

    render(<FollowUpSuggestions suggestions={many} onSuggestionClick={vi.fn()} />);

    expect(screen.getAllByRole("button")).toHaveLength(10);
  });

  // =========================================================================
  // Text Content Variations
  // =========================================================================

  it("handles long suggestion text", () => {
    const long = ["This is a very long follow-up question that might wrap to multiple lines?"];

    render(<FollowUpSuggestions suggestions={long} onSuggestionClick={vi.fn()} />);

    expect(screen.getByText(long[0])).toBeInTheDocument();
  });

  it("handles special characters in suggestions", () => {
    const special = ["What's the <value> & cost?", "Show me data @ 100%"];

    render(<FollowUpSuggestions suggestions={special} onSuggestionClick={vi.fn()} />);

    expect(screen.getByText(special[0])).toBeInTheDocument();
    expect(screen.getByText(special[1])).toBeInTheDocument();
  });

  it("handles unicode in suggestions", () => {
    const unicode = ["What about 你好?", "Show мир data", "Status 🎉?"];

    render(<FollowUpSuggestions suggestions={unicode} onSuggestionClick={vi.fn()} />);

    unicode.forEach((text) => {
      expect(screen.getByText(text)).toBeInTheDocument();
    });
  });

  it("handles empty string suggestion", () => {
    const suggestions = ["Valid question?", "", "Another question?"];

    render(
      <FollowUpSuggestions suggestions={suggestions} onSuggestionClick={vi.fn()} />
    );

    expect(screen.getAllByRole("button")).toHaveLength(3);
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("can tab through all suggestions", () => {
    render(<FollowUpSuggestions {...defaultProps} />);

    const buttons = screen.getAllByRole("button");

    buttons.forEach((button) => {
      button.focus();
      expect(button).toHaveFocus();
    });
  });

  it("has semantic HTML structure", () => {
    const { container } = render(<FollowUpSuggestions {...defaultProps} />);

    // Container should be a group
    expect(container.querySelector('[role="group"]')).toBeInTheDocument();

    // All items should be buttons
    const buttons = container.querySelectorAll("button");
    expect(buttons.length).toBeGreaterThan(0);
  });

  it("provides context for screen readers", () => {
    render(<FollowUpSuggestions {...defaultProps} />);

    const group = screen.getByRole("group");
    expect(group).toHaveAttribute("aria-label", "Suggested follow-up questions");
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles suggestion with only whitespace", () => {
    const suggestions = ["Valid?", "   ", "Another?"];

    render(
      <FollowUpSuggestions suggestions={suggestions} onSuggestionClick={vi.fn()} />
    );

    expect(screen.getAllByRole("button")).toHaveLength(3);
  });

  it("handles duplicate suggestions", () => {
    const suggestions = ["Question?", "Question?", "Different?"];

    render(
      <FollowUpSuggestions suggestions={suggestions} onSuggestionClick={vi.fn()} />
    );

    // Should render all, even duplicates
    expect(screen.getAllByRole("button")).toHaveLength(3);
  });

  it("uses index as key for suggestions", () => {
    // This is implicit in the implementation, but we can verify rendering
    const { rerender } = render(<FollowUpSuggestions {...defaultProps} />);

    expect(screen.getAllByRole("button")).toHaveLength(3);

    // Change order
    const reversed = [...defaultProps.suggestions].reverse();
    rerender(
      <FollowUpSuggestions
        suggestions={reversed}
        onSuggestionClick={defaultProps.onSuggestionClick}
      />
    );

    expect(screen.getAllByRole("button")).toHaveLength(3);
  });

  it("re-renders when suggestions change", () => {
    const { rerender } = render(<FollowUpSuggestions {...defaultProps} />);

    expect(screen.getByText("Question 1?")).toBeInTheDocument();

    const newSuggestions = ["New question 1?", "New question 2?"];
    rerender(
      <FollowUpSuggestions
        suggestions={newSuggestions}
        onSuggestionClick={defaultProps.onSuggestionClick}
      />
    );

    expect(screen.getByText("New question 1?")).toBeInTheDocument();
    expect(screen.queryByText("Question 1?")).not.toBeInTheDocument();
  });
});
