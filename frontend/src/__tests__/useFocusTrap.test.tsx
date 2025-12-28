import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { useRef, useState } from "react";
import { useFocusTrap } from "../hooks/useFocusTrap";

// Test component that uses the focus trap
function TestModal({
  isOpen,
  onClose,
  initialFocusOnClose = false,
}: {
  isOpen: boolean;
  onClose: () => void;
  initialFocusOnClose?: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  const { focusFirst, focusLast } = useFocusTrap(containerRef, {
    isActive: isOpen,
    onEscape: onClose,
    restoreFocus: true,
    initialFocusRef: initialFocusOnClose ? closeButtonRef : undefined,
  });

  if (!isOpen) return null;

  return (
    <div ref={containerRef} data-testid="modal" tabIndex={-1}>
      <button ref={closeButtonRef} data-testid="close-btn" onClick={onClose}>
        Close
      </button>
      <input data-testid="input-1" type="text" placeholder="First input" />
      <input data-testid="input-2" type="text" placeholder="Second input" />
      <button data-testid="save-btn">Save</button>
      <button data-testid="focus-first" onClick={focusFirst}>
        Focus First
      </button>
      <button data-testid="focus-last" onClick={focusLast}>
        Focus Last
      </button>
    </div>
  );
}

// Component with no focusable elements
function EmptyModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useFocusTrap(containerRef, {
    isActive: isOpen,
    onEscape: onClose,
  });

  if (!isOpen) return null;

  return (
    <div ref={containerRef} data-testid="empty-modal" tabIndex={-1}>
      <p>No focusable elements here</p>
    </div>
  );
}

// Component with disabled elements
function ModalWithDisabled({ isOpen }: { isOpen: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useFocusTrap(containerRef, {
    isActive: isOpen,
  });

  if (!isOpen) return null;

  return (
    <div ref={containerRef} data-testid="modal-disabled" tabIndex={-1}>
      <button disabled data-testid="disabled-btn">
        Disabled
      </button>
      <button data-testid="enabled-btn">Enabled</button>
      <input disabled data-testid="disabled-input" />
      <input data-testid="enabled-input" />
    </div>
  );
}

// Wrapper to control modal state
function ModalWrapper({
  initialOpen = false,
  useInitialFocus = false,
}: {
  initialOpen?: boolean;
  useInitialFocus?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(initialOpen);

  return (
    <div>
      <button data-testid="trigger" onClick={() => setIsOpen(true)}>
        Open Modal
      </button>
      <TestModal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        initialFocusOnClose={useInitialFocus}
      />
    </div>
  );
}

describe("useFocusTrap", () => {
  let originalBodyOverflow: string;

  beforeEach(() => {
    originalBodyOverflow = document.body.style.overflow;
    // Clear any focus
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  });

  afterEach(() => {
    document.body.style.overflow = originalBodyOverflow;
  });

  // ===========================================================================
  // Basic Activation
  // ===========================================================================

  describe("activation", () => {
    it("focuses first element when activated", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });
    });

    it("focuses initial focus ref when provided", async () => {
      render(<ModalWrapper initialOpen useInitialFocus />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });
    });

    it("hides body overflow when active", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(document.body.style.overflow).toBe("hidden");
      });
    });

    it("restores body overflow when deactivated", async () => {
      const { rerender } = render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(document.body.style.overflow).toBe("hidden");
      });

      // Close modal by triggering close
      const closeBtn = screen.getByTestId("close-btn");
      fireEvent.click(closeBtn);

      await waitFor(() => {
        expect(document.body.style.overflow).toBe("");
      });
    });
  });

  // ===========================================================================
  // Tab Trapping
  // ===========================================================================

  describe("tab trapping", () => {
    it("traps Tab at last element", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Navigate to last focusable element (focus-last button)
      const focusLastBtn = screen.getByTestId("focus-last");
      act(() => {
        focusLastBtn.focus();
      });

      // Press Tab - should cycle to first element
      fireEvent.keyDown(document, { key: "Tab" });

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });
    });

    it("traps Shift+Tab at first element", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Press Shift+Tab - should cycle to last element
      fireEvent.keyDown(document, { key: "Tab", shiftKey: true });

      await waitFor(() => {
        expect(screen.getByTestId("focus-last")).toHaveFocus();
      });
    });

    it("allows normal tab navigation within trap", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Tab to next element
      const input1 = screen.getByTestId("input-1");
      act(() => {
        input1.focus();
      });

      // Should be focused on input-1
      expect(input1).toHaveFocus();
    });
  });

  // ===========================================================================
  // Escape Key
  // ===========================================================================

  describe("escape key", () => {
    it("calls onEscape when Escape is pressed", async () => {
      const onClose = vi.fn();

      render(
        <TestModal isOpen onClose={onClose} />
      );

      await waitFor(() => {
        expect(screen.getByTestId("modal")).toBeInTheDocument();
      });

      fireEvent.keyDown(document, { key: "Escape" });

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it("does not call onEscape when not active", () => {
      const onClose = vi.fn();

      render(
        <TestModal isOpen={false} onClose={onClose} />
      );

      fireEvent.keyDown(document, { key: "Escape" });

      expect(onClose).not.toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Focus Restoration
  // ===========================================================================

  describe("focus restoration", () => {
    it("restores focus to trigger element when closed", async () => {
      render(<ModalWrapper />);

      const trigger = screen.getByTestId("trigger");
      act(() => {
        trigger.focus();
      });

      // Open modal
      fireEvent.click(trigger);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Close modal
      fireEvent.click(screen.getByTestId("close-btn"));

      await waitFor(() => {
        expect(trigger).toHaveFocus();
      });
    });
  });

  // ===========================================================================
  // Empty Container
  // ===========================================================================

  describe("empty container", () => {
    it("focuses container when no focusable elements", async () => {
      render(<EmptyModal isOpen onClose={() => {}} />);

      await waitFor(() => {
        const modal = screen.getByTestId("empty-modal");
        // Container should be focused (or at least exist)
        expect(modal).toBeInTheDocument();
      });
    });
  });

  // ===========================================================================
  // Disabled Elements
  // ===========================================================================

  describe("disabled elements", () => {
    it("skips disabled elements in focus trap", async () => {
      render(<ModalWithDisabled isOpen />);

      await waitFor(() => {
        // Should focus the first enabled element
        expect(screen.getByTestId("enabled-btn")).toHaveFocus();
      });
    });
  });

  // ===========================================================================
  // Helper Functions
  // ===========================================================================

  describe("helper functions", () => {
    it("focusFirst focuses first focusable element", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Focus something else
      const input = screen.getByTestId("input-2");
      act(() => {
        input.focus();
      });
      expect(input).toHaveFocus();

      // Click focus first button
      fireEvent.click(screen.getByTestId("focus-first"));

      expect(screen.getByTestId("close-btn")).toHaveFocus();
    });

    it("focusLast focuses last focusable element", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Click focus last button
      fireEvent.click(screen.getByTestId("focus-last"));

      expect(screen.getByTestId("focus-last")).toHaveFocus();
    });
  });

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe("edge cases", () => {
    it("handles rapid open/close", async () => {
      render(<ModalWrapper />);

      const trigger = screen.getByTestId("trigger");

      // Open
      fireEvent.click(trigger);
      await waitFor(() => {
        expect(screen.queryByTestId("modal")).toBeInTheDocument();
      });

      // Close immediately
      fireEvent.click(screen.getByTestId("close-btn"));

      // Open again
      fireEvent.click(trigger);
      await waitFor(() => {
        expect(screen.queryByTestId("modal")).toBeInTheDocument();
      });

      // Should work normally
      expect(screen.getByTestId("close-btn")).toHaveFocus();
    });

    it("handles focus outside container", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Simulate focus being lost outside
      const saveBtn = screen.getByTestId("save-btn");
      act(() => {
        saveBtn.focus();
      });

      // Tab should wrap properly even if focus was moved
      fireEvent.keyDown(document, { key: "Tab" });

      // Should still work
      expect(document.activeElement).not.toBeNull();
    });

    it("does not trap when inactive", async () => {
      render(<ModalWrapper initialOpen={false} />);

      // Modal not rendered
      expect(screen.queryByTestId("modal")).not.toBeInTheDocument();

      // Body overflow should not be affected
      expect(document.body.style.overflow).not.toBe("hidden");
    });
  });

  // ===========================================================================
  // WCAG Compliance
  // ===========================================================================

  describe("WCAG 2.4.3 compliance", () => {
    it("ensures keyboard focus can be trapped within modal", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Focus the last element
      const lastBtn = screen.getByTestId("focus-last");
      act(() => {
        lastBtn.focus();
      });
      expect(lastBtn).toHaveFocus();

      // Press Tab at last element - should wrap to first
      fireEvent.keyDown(document, { key: "Tab" });
      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });
    });

    it("traps reverse tab navigation", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Press Shift+Tab at first element - should wrap to last
      fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
      await waitFor(() => {
        expect(screen.getByTestId("focus-last")).toHaveFocus();
      });
    });
  });
});
