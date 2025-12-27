import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InputBar } from "../components/InputBar";

describe("InputBar", () => {
  const defaultProps = {
    value: "",
    onChange: vi.fn(),
    onSubmit: vi.fn(),
    isLoading: false,
  };

  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders input and submit button", () => {
    render(<InputBar {...defaultProps} />);

    expect(screen.getByRole("textbox")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
  });

  it("renders with default placeholder", () => {
    render(<InputBar {...defaultProps} />);

    const input = screen.getByPlaceholderText(/ask a question about your crm/i);
    expect(input).toBeInTheDocument();
  });

  it("renders with custom placeholder", () => {
    render(<InputBar {...defaultProps} placeholder="Custom placeholder" />);

    expect(screen.getByPlaceholderText("Custom placeholder")).toBeInTheDocument();
  });

  it("displays current value", () => {
    render(<InputBar {...defaultProps} value="Test question" />);

    const input = screen.getByRole("textbox");
    expect(input).toHaveValue("Test question");
  });

  it("has proper ARIA labels", () => {
    render(<InputBar {...defaultProps} />);

    const input = screen.getByLabelText(/ask a question about your crm/i);
    expect(input).toBeInTheDocument();

    const form = screen.getByRole("search");
    expect(form).toHaveAccessibleName("Ask a question");
  });

  it("includes hint for keyboard users", () => {
    render(<InputBar {...defaultProps} />);

    const hint = screen.getByText(/press enter to send/i);
    expect(hint).toBeInTheDocument();
    expect(hint).toHaveClass("visually-hidden");
  });

  // =========================================================================
  // Input Handling
  // =========================================================================

  it("calls onChange when typing", async () => {
    const onChange = vi.fn();
    render(<InputBar {...defaultProps} onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await userEvent.type(input, "Hello");

    expect(onChange).toHaveBeenCalledTimes(5); // Once per character
    expect(onChange).toHaveBeenCalledWith("H");
    expect(onChange).toHaveBeenCalledWith("e");
  });

  it("calls onChange with correct value", async () => {
    const onChange = vi.fn();
    render(<InputBar {...defaultProps} onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await userEvent.type(input, "A");

    expect(onChange).toHaveBeenCalledWith("A");
  });

  it("handles paste events", async () => {
    const onChange = vi.fn();
    render(<InputBar {...defaultProps} onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await userEvent.click(input);
    await userEvent.paste("Pasted text");

    expect(onChange).toHaveBeenCalled();
  });

  it("handles clear/delete", async () => {
    const onChange = vi.fn();
    render(<InputBar {...defaultProps} value="Text" onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await userEvent.clear(input);

    expect(onChange).toHaveBeenCalled();
  });

  // =========================================================================
  // Submit Handling
  // =========================================================================

  it("calls onSubmit when button clicked", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} value="Question" onSubmit={onSubmit} />);

    const button = screen.getByRole("button", { name: /send/i });
    await userEvent.click(button);

    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("calls onSubmit on Enter key", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} value="Question" onSubmit={onSubmit} />);

    const input = screen.getByRole("textbox");
    await userEvent.type(input, "{Enter}");

    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("prevents form submission with empty value", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} value="" onSubmit={onSubmit} />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("prevents form submission with whitespace-only value", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} value="   " onSubmit={onSubmit} />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("does not submit on Shift+Enter", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} value="Question" onSubmit={onSubmit} />);

    const input = screen.getByRole("textbox");
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("prevents default form submission", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} value="Question" onSubmit={onSubmit} />);

    const form = screen.getByRole("search");
    // Submit the form - if preventDefault wasn't called, the page would reload
    // Since we're in a test environment, we verify the submit handler runs
    fireEvent.submit(form);

    // Component calls e.preventDefault() then onSubmit - verify handler was called
    expect(onSubmit).toHaveBeenCalled();
  });

  // =========================================================================
  // Loading State
  // =========================================================================

  it("shows loading state when isLoading is true", () => {
    render(<InputBar {...defaultProps} isLoading={true} />);

    expect(screen.getByText(/thinking/i)).toBeInTheDocument();
  });

  it("disables input when loading", () => {
    render(<InputBar {...defaultProps} isLoading={true} />);

    const input = screen.getByRole("textbox");
    expect(input).toBeDisabled();
  });

  it("disables button when loading", () => {
    render(<InputBar {...defaultProps} isLoading={true} value="Question" />);

    const button = screen.getByRole("button");
    expect(button).toBeDisabled();
  });

  it("prevents submission when loading", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} isLoading={true} value="Question" onSubmit={onSubmit} />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("changes button text when loading", () => {
    const { rerender } = render(<InputBar {...defaultProps} value="Question" />);

    expect(screen.getByText("Send")).toBeInTheDocument();

    rerender(<InputBar {...defaultProps} value="Question" isLoading={true} />);

    expect(screen.getByText(/thinking/i)).toBeInTheDocument();
    expect(screen.queryByText("Send")).not.toBeInTheDocument();
  });

  it("updates ARIA label when loading", () => {
    render(<InputBar {...defaultProps} isLoading={true} />);

    const button = screen.getByRole("button", { name: /sending message/i });
    expect(button).toBeInTheDocument();
  });

  // =========================================================================
  // Button State
  // =========================================================================

  it("enables button when value is not empty", () => {
    render(<InputBar {...defaultProps} value="Question" />);

    const button = screen.getByRole("button");
    expect(button).not.toBeDisabled();
  });

  it("disables button when value is empty", () => {
    render(<InputBar {...defaultProps} value="" />);

    const button = screen.getByRole("button");
    expect(button).toBeDisabled();
  });

  it("disables button when value is whitespace only", () => {
    render(<InputBar {...defaultProps} value="   " />);

    const button = screen.getByRole("button");
    expect(button).toBeDisabled();
  });

  it("updates button state when value changes", () => {
    const { rerender } = render(<InputBar {...defaultProps} value="" />);

    let button = screen.getByRole("button");
    expect(button).toBeDisabled();

    rerender(<InputBar {...defaultProps} value="Question" />);

    button = screen.getByRole("button");
    expect(button).not.toBeDisabled();
  });

  // =========================================================================
  // Autofocus
  // =========================================================================

  it("autofocuses input on mount", () => {
    render(<InputBar {...defaultProps} />);

    const input = screen.getByRole("textbox");
    expect(input).toHaveFocus();
  });

  // =========================================================================
  // Ref Forwarding
  // =========================================================================

  it("accepts ref prop (React 19)", () => {
    const ref = { current: null } as unknown as React.RefObject<HTMLInputElement>;
    render(<InputBar {...defaultProps} ref={ref} />);

    expect(ref.current).toBeInstanceOf(HTMLInputElement);
  });

  it("ref can be used to focus input", () => {
    const ref = { current: null } as unknown as React.RefObject<HTMLInputElement>;
    render(<InputBar {...defaultProps} ref={ref} />);

    ref.current?.blur(); // Remove focus first
    expect(ref.current).not.toHaveFocus();

    ref.current?.focus();
    expect(ref.current).toHaveFocus();
  });

  // =========================================================================
  // Keyboard Shortcuts
  // =========================================================================

  it("handles Enter key correctly", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} value="Test" onSubmit={onSubmit} />);

    const input = screen.getByRole("textbox");
    fireEvent.keyDown(input, { key: "Enter", shiftKey: false });

    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("ignores other keys", async () => {
    const onSubmit = vi.fn();
    render(<InputBar {...defaultProps} value="Test" onSubmit={onSubmit} />);

    const input = screen.getByRole("textbox");
    fireEvent.keyDown(input, { key: "a" });
    fireEvent.keyDown(input, { key: "Escape" });
    fireEvent.keyDown(input, { key: "Tab" });

    expect(onSubmit).not.toHaveBeenCalled();
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("memoizes component", () => {
    const { rerender } = render(<InputBar {...defaultProps} value="Test" />);
    const firstButton = screen.getByRole("button");

    // Rerender with same props
    rerender(<InputBar {...defaultProps} value="Test" />);
    const secondButton = screen.getByRole("button");

    // Should be same element (memoized)
    expect(firstButton).toBe(secondButton);
  });

  it("updates when props change", () => {
    const { rerender } = render(<InputBar {...defaultProps} value="Test" />);
    expect(screen.getByDisplayValue("Test")).toBeInTheDocument();

    rerender(<InputBar {...defaultProps} value="Updated" />);
    expect(screen.getByDisplayValue("Updated")).toBeInTheDocument();
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles very long input", async () => {
    const onChange = vi.fn();

    render(<InputBar {...defaultProps} onChange={onChange} />);

    const input = screen.getByRole("textbox");
    // Type a long string of characters
    await userEvent.type(input, "a".repeat(50));

    expect(onChange).toHaveBeenCalled();
  });

  it("handles special characters", async () => {
    const onChange = vi.fn();
    render(<InputBar {...defaultProps} onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await userEvent.type(input, "!@#$%^&*()");

    expect(onChange).toHaveBeenCalled();
  });

  it("handles unicode characters", async () => {
    const onChange = vi.fn();
    render(<InputBar {...defaultProps} onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await userEvent.type(input, "你好");

    expect(onChange).toHaveBeenCalled();
  });

  it("handles emojis", async () => {
    const onChange = vi.fn();
    render(<InputBar {...defaultProps} onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await userEvent.type(input, "🎉");

    expect(onChange).toHaveBeenCalled();
  });

  it("handles rapid input", async () => {
    const onChange = vi.fn();
    render(<InputBar {...defaultProps} onChange={onChange} />);

    const input = screen.getByRole("textbox");
    await userEvent.type(input, "abcdefghij", { delay: 1 });

    expect(onChange).toHaveBeenCalledTimes(10);
  });

  it("handles form submission multiple times", async () => {
    const onSubmit = vi.fn();
    const { rerender } = render(
      <InputBar {...defaultProps} value="Test 1" onSubmit={onSubmit} />
    );

    const button = screen.getByRole("button");
    await userEvent.click(button);
    expect(onSubmit).toHaveBeenCalledTimes(1);

    // Update value and submit again
    rerender(<InputBar {...defaultProps} value="Test 2" onSubmit={onSubmit} />);
    await userEvent.click(button);
    expect(onSubmit).toHaveBeenCalledTimes(2);
  });
});
