import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CopyButton } from "../components/CopyButton";

// Mock clipboard API
const mockWriteText = vi.fn();
Object.defineProperty(navigator, 'clipboard', {
  value: {
    writeText: mockWriteText,
  },
  writable: true,
  configurable: true,
});

describe("CopyButton", () => {
  beforeEach(() => {
    mockWriteText.mockReset();
    mockWriteText.mockResolvedValue(undefined);
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders copy button", () => {
    render(<CopyButton text="Test text" />);

    const button = screen.getByRole("button", { name: /copy to clipboard/i });
    expect(button).toBeInTheDocument();
  });

  it("has copy icon initially", () => {
    const { container } = render(<CopyButton text="Test" />);

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("applies custom className", () => {
    const { container } = render(<CopyButton text="Test" className="custom" />);

    const button = container.querySelector("button");
    expect(button).toHaveClass("copy-button", "custom");
  });

  it("has proper ARIA attributes", () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-label", "Copy to clipboard");
    expect(button).toHaveAttribute("title", "Copy to clipboard");
  });

  it("is a button type", () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("type", "button");
  });

  // =========================================================================
  // Copy Functionality
  // =========================================================================

  it("copies text to clipboard on click", async () => {
    render(<CopyButton text="Test content" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(mockWriteText).toHaveBeenCalledWith("Test content");
  });

  it("shows copied state after successful copy", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    await waitFor(() => {
      expect(button).toHaveAttribute("aria-label", "Copied!");
      expect(button).toHaveClass("copy-button--copied");
    });
  });

  it("shows checkmark icon when copied", async () => {
    const { container } = render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    await waitFor(() => {
      // Checkmark path has specific d attribute
      const checkmark = container.querySelector('path[d*="16.17"]');
      expect(checkmark).toBeInTheDocument();
    });
  });

  it("resets copied state after timeout", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    // Initially copied
    await waitFor(() => {
      expect(button).toHaveAttribute("aria-label", "Copied!");
    });

    // Fast-forward 2 seconds
    vi.advanceTimersByTime(2000);

    await waitFor(() => {
      expect(button).toHaveAttribute("aria-label", "Copy to clipboard");
    });
  });

  it("removes copied class after timeout", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    await waitFor(() => {
      expect(button).toHaveClass("copy-button--copied");
    });

    vi.advanceTimersByTime(2000);

    await waitFor(() => {
      expect(button).not.toHaveClass("copy-button--copied");
    });
  });

  // =========================================================================
  // Error Handling
  // =========================================================================

  it("handles clipboard write failure gracefully", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    mockWriteText.mockRejectedValue(new Error("Permission denied"));

    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    await waitFor(() => {
      expect(consoleError).toHaveBeenCalledWith(
        "Failed to copy:",
        expect.any(Error)
      );
    });

    consoleError.mockRestore();
  });

  it("does not show copied state on error", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mockWriteText.mockRejectedValue(new Error("Failed"));

    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    await waitFor(() => {
      expect(button).toHaveAttribute("aria-label", "Copy to clipboard");
      expect(button).not.toHaveClass("copy-button--copied");
    });
  });

  // =========================================================================
  // Multiple Clicks
  // =========================================================================

  it("handles multiple clicks", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");

    await userEvent.click(button);
    expect(mockWriteText).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(2000);

    await userEvent.click(button);
    expect(mockWriteText).toHaveBeenCalledTimes(2);
  });

  it("resets timeout on multiple rapid clicks", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");

    // First click
    await userEvent.click(button);
    await waitFor(() => expect(button).toHaveClass("copy-button--copied"));

    // Click again before timeout
    vi.advanceTimersByTime(1000);
    await userEvent.click(button);

    // Should still be in copied state
    expect(button).toHaveClass("copy-button--copied");

    // Wait for new timeout
    vi.advanceTimersByTime(2000);

    await waitFor(() => {
      expect(button).not.toHaveClass("copy-button--copied");
    });
  });

  // =========================================================================
  // Different Text Content
  // =========================================================================

  it("copies empty string", async () => {
    render(<CopyButton text="" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(mockWriteText).toHaveBeenCalledWith("");
  });

  it("copies long text", async () => {
    const longText = "a".repeat(10000);
    render(<CopyButton text={longText} />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(mockWriteText).toHaveBeenCalledWith(longText);
  });

  it("copies text with special characters", async () => {
    const specialText = "Test: <>&\"'\n\t@#$%";
    render(<CopyButton text={specialText} />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(mockWriteText).toHaveBeenCalledWith(specialText);
  });

  it("copies unicode text", async () => {
    const unicodeText = "Hello 你好 мир 🎉";
    render(<CopyButton text={unicodeText} />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(mockWriteText).toHaveBeenCalledWith(unicodeText);
  });

  it("copies markdown text", async () => {
    const markdownText = "# Header\n**bold** and *italic*\n- List item";
    render(<CopyButton text={markdownText} />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    expect(mockWriteText).toHaveBeenCalledWith(markdownText);
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("memoizes component", () => {
    const { rerender } = render(<CopyButton text="Test" />);
    const firstButton = screen.getByRole("button");

    rerender(<CopyButton text="Test" />);
    const secondButton = screen.getByRole("button");

    expect(firstButton).toBe(secondButton);
  });

  it("re-renders when text changes", async () => {
    const { rerender } = render(<CopyButton text="Original" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);
    expect(mockWriteText).toHaveBeenLastCalledWith("Original");

    rerender(<CopyButton text="Updated" />);

    // New text should be copied
    await userEvent.click(button);
    expect(mockWriteText).toHaveBeenLastCalledWith("Updated");
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("is keyboard accessible", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    button.focus();

    expect(button).toHaveFocus();
  });

  it("can be activated with keyboard", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    button.focus();

    // Simulate Enter key
    await userEvent.keyboard("{Enter}");

    expect(mockWriteText).toHaveBeenCalled();
  });

  it("updates ARIA label when state changes", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");

    expect(button).toHaveAttribute("aria-label", "Copy to clipboard");

    await userEvent.click(button);

    await waitFor(() => {
      expect(button).toHaveAttribute("aria-label", "Copied!");
    });

    vi.advanceTimersByTime(2000);

    await waitFor(() => {
      expect(button).toHaveAttribute("aria-label", "Copy to clipboard");
    });
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles component unmount during timeout", async () => {
    const { unmount } = render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");
    await userEvent.click(button);

    // Unmount before timeout
    unmount();

    // Should not throw error
    vi.advanceTimersByTime(2000);
  });

  it("handles rapid clicks with timer", async () => {
    render(<CopyButton text="Test" />);

    const button = screen.getByRole("button");

    for (let i = 0; i < 5; i++) {
      await userEvent.click(button);
      vi.advanceTimersByTime(500);
    }

    expect(mockWriteText).toHaveBeenCalled();
  });
});
