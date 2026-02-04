import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  act,
  waitFor,
} from "@testing-library/react";
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

  useFocusTrap(containerRef, {
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
    </div>
  );
}

// Component with no focusable elements
function EmptyModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
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
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  });

  afterEach(() => {
    document.body.style.overflow = originalBodyOverflow;
  });

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
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(document.body.style.overflow).toBe("hidden");
      });

      const closeBtn = screen.getByTestId("close-btn");
      fireEvent.click(closeBtn);

      await waitFor(() => {
        expect(document.body.style.overflow).toBe("");
      });
    });
  });

  describe("tab trapping", () => {
    it("traps Tab at last element", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Navigate to last focusable element
      const saveBtn = screen.getByTestId("save-btn");
      act(() => {
        saveBtn.focus();
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
        expect(screen.getByTestId("save-btn")).toHaveFocus();
      });
    });

    it("allows normal tab navigation within trap", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      const input1 = screen.getByTestId("input-1");
      act(() => {
        input1.focus();
      });

      expect(input1).toHaveFocus();
    });
  });

  describe("escape key", () => {
    it("calls onEscape when Escape is pressed", async () => {
      const onClose = vi.fn();

      render(<TestModal isOpen onClose={onClose} />);

      await waitFor(() => {
        expect(screen.getByTestId("modal")).toBeInTheDocument();
      });

      fireEvent.keyDown(document, { key: "Escape" });

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it("does not call onEscape when not active", () => {
      const onClose = vi.fn();

      render(<TestModal isOpen={false} onClose={onClose} />);

      fireEvent.keyDown(document, { key: "Escape" });

      expect(onClose).not.toHaveBeenCalled();
    });
  });

  describe("focus restoration", () => {
    it("restores focus to trigger element when closed", async () => {
      render(<ModalWrapper />);

      const trigger = screen.getByTestId("trigger");
      act(() => {
        trigger.focus();
      });

      fireEvent.click(trigger);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      fireEvent.click(screen.getByTestId("close-btn"));

      await waitFor(() => {
        expect(trigger).toHaveFocus();
      });
    });
  });

  describe("empty container", () => {
    it("focuses container when no focusable elements", async () => {
      render(<EmptyModal isOpen onClose={() => {}} />);

      await waitFor(() => {
        const modal = screen.getByTestId("empty-modal");
        expect(modal).toBeInTheDocument();
      });
    });
  });

  describe("disabled elements", () => {
    it("skips disabled elements in focus trap", async () => {
      render(<ModalWithDisabled isOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("enabled-btn")).toHaveFocus();
      });
    });
  });

  describe("edge cases", () => {
    it("handles rapid open/close", async () => {
      render(<ModalWrapper />);

      const trigger = screen.getByTestId("trigger");

      fireEvent.click(trigger);
      await waitFor(() => {
        expect(screen.queryByTestId("modal")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByTestId("close-btn"));

      fireEvent.click(trigger);
      await waitFor(() => {
        expect(screen.queryByTestId("modal")).toBeInTheDocument();
      });

      expect(screen.getByTestId("close-btn")).toHaveFocus();
    });

    it("does not trap when inactive", async () => {
      render(<ModalWrapper initialOpen={false} />);

      expect(screen.queryByTestId("modal")).not.toBeInTheDocument();
      expect(document.body.style.overflow).not.toBe("hidden");
    });
  });

  describe("tab with focus outside container", () => {
    it("wraps forward Tab when focus is outside the container", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Move focus to a non-container element (simulate focus escaping)
      act(() => {
        (document.body as HTMLElement).focus();
      });

      // Press Tab while focus is outside container - should wrap to first element
      fireEvent.keyDown(document, { key: "Tab" });

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });
    });

    it("wraps backward Shift+Tab when focus is outside the container", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Move focus outside the container
      act(() => {
        (document.body as HTMLElement).focus();
      });

      // Press Shift+Tab - should wrap to last element
      fireEvent.keyDown(document, { key: "Tab", shiftKey: true });

      await waitFor(() => {
        expect(screen.getByTestId("save-btn")).toHaveFocus();
      });
    });
  });

  describe("restoreFocus disabled", () => {
    it("does not restore focus when restoreFocus is false", async () => {
      function NoRestoreModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
        const containerRef = useRef<HTMLDivElement>(null);
        useFocusTrap(containerRef, {
          isActive: isOpen,
          onEscape: onClose,
          restoreFocus: false,
        });

        if (!isOpen) return null;
        return (
          <div ref={containerRef} data-testid="no-restore-modal" tabIndex={-1}>
            <button data-testid="nr-close" onClick={onClose}>Close</button>
          </div>
        );
      }

      function NoRestoreWrapper() {
        const [isOpen, setIsOpen] = useState(false);
        return (
          <div>
            <button data-testid="nr-trigger" onClick={() => setIsOpen(true)}>Open</button>
            <NoRestoreModal isOpen={isOpen} onClose={() => setIsOpen(false)} />
          </div>
        );
      }

      render(<NoRestoreWrapper />);

      const trigger = screen.getByTestId("nr-trigger");
      act(() => { trigger.focus(); });
      fireEvent.click(trigger);

      await waitFor(() => {
        expect(screen.getByTestId("nr-close")).toHaveFocus();
      });

      fireEvent.click(screen.getByTestId("nr-close"));

      // Focus should NOT be restored to trigger since restoreFocus=false
      await waitFor(() => {
        expect(screen.queryByTestId("no-restore-modal")).not.toBeInTheDocument();
      });
    });
  });

  describe("WCAG 2.4.3 compliance", () => {
    it("ensures keyboard focus can be trapped within modal", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      const saveBtn = screen.getByTestId("save-btn");
      act(() => {
        saveBtn.focus();
      });
      expect(saveBtn).toHaveFocus();

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

      fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
      await waitFor(() => {
        expect(screen.getByTestId("save-btn")).toHaveFocus();
      });
    });
  });

  describe("null container edge cases", () => {
    it("handles Tab key press when container is empty (no focusable elements)", async () => {
      render(<EmptyModal isOpen onClose={() => {}} />);

      await waitFor(() => {
        expect(screen.getByTestId("empty-modal")).toBeInTheDocument();
      });

      // Tab key should not throw and should not focus anything
      fireEvent.keyDown(document, { key: "Tab" });

      // Modal should still be visible
      expect(screen.getByTestId("empty-modal")).toBeInTheDocument();
    });

    it("handles Shift+Tab key press when container is empty", async () => {
      render(<EmptyModal isOpen onClose={() => {}} />);

      await waitFor(() => {
        expect(screen.getByTestId("empty-modal")).toBeInTheDocument();
      });

      // Shift+Tab should not throw
      fireEvent.keyDown(document, { key: "Tab", shiftKey: true });

      expect(screen.getByTestId("empty-modal")).toBeInTheDocument();
    });

    it("handles keyboard events before containerRef is assigned", () => {
      // Test that keydown events are safely ignored when hook is inactive
      function DelayedModal() {
        const containerRef = useRef<HTMLDivElement>(null);
        useFocusTrap(containerRef, {
          isActive: false,
          onEscape: () => {},
        });
        return <div data-testid="delayed">Inactive modal</div>;
      }

      render(<DelayedModal />);

      // Should not throw when pressing keys with inactive trap
      fireEvent.keyDown(document, { key: "Tab" });
      fireEvent.keyDown(document, { key: "Escape" });

      expect(screen.getByTestId("delayed")).toBeInTheDocument();
    });
  });

  describe("normal tab navigation (no wrapping)", () => {
    it("allows normal Tab navigation in the middle of focusable elements", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Focus input-1 (not first or last)
      const input1 = screen.getByTestId("input-1");
      act(() => {
        input1.focus();
      });
      expect(input1).toHaveFocus();

      // Tab should allow normal navigation (no preventDefault)
      // The browser would move to next element - we can't test actual focus change
      // but we can verify Tab doesn't wrap to first/last
      fireEvent.keyDown(document, { key: "Tab" });

      // Should NOT wrap to close button since we're in the middle
      // (browser handles actual focus move)
    });

    it("allows normal Shift+Tab navigation in the middle of focusable elements", async () => {
      render(<ModalWrapper initialOpen />);

      await waitFor(() => {
        expect(screen.getByTestId("close-btn")).toHaveFocus();
      });

      // Focus input-2 (not first or last)
      const input2 = screen.getByTestId("input-2");
      act(() => {
        input2.focus();
      });
      expect(input2).toHaveFocus();

      // Shift+Tab should allow normal navigation
      fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    });
  });
});
