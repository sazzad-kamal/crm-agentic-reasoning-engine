import { useEffect, useRef, useCallback, RefObject } from "react";

/**
 * Focusable element selectors for trap management
 */
const FOCUSABLE_SELECTORS = [
  'button:not([disabled])',
  'a[href]',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

interface UseFocusTrapOptions {
  /** Whether the focus trap is active */
  isActive: boolean;
  /** Callback when escape key is pressed */
  onEscape?: () => void;
  /** Whether to restore focus when deactivated */
  restoreFocus?: boolean;
  /** Initial element to focus when activated */
  initialFocusRef?: RefObject<HTMLElement | null>;
}

/**
 * Hook to trap focus within a container element
 * Essential for modal accessibility (WCAG 2.4.3)
 *
 * @param containerRef - Ref to the container element to trap focus within
 * @param options - Configuration options
 *
 * @example
 * ```tsx
 * const drawerRef = useRef<HTMLDivElement>(null);
 * useFocusTrap(drawerRef, { isActive: isDrawerOpen, onEscape: closeDrawer });
 * ```
 */
export function useFocusTrap(
  containerRef: RefObject<HTMLElement | null>,
  options: UseFocusTrapOptions
) {
  const { isActive, onEscape, restoreFocus = true, initialFocusRef } = options;
  const previousActiveElement = useRef<HTMLElement | null>(null);

  /**
   * Get all focusable elements within the container
   */
  const getFocusableElements = useCallback((): HTMLElement[] => {
    if (!containerRef.current) return [];
    const elements = containerRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS);
    return Array.from(elements).filter(
      (el) => el.offsetParent !== null && !el.hasAttribute('inert')
    );
  }, [containerRef]);

  /**
   * Handle tab key to trap focus
   */
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!isActive || !containerRef.current) return;

      // Handle Escape key
      if (event.key === "Escape" && onEscape) {
        event.preventDefault();
        onEscape();
        return;
      }

      // Handle Tab key for focus trapping
      if (event.key === "Tab") {
        const focusableElements = getFocusableElements();
        if (focusableElements.length === 0) return;

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        const activeElement = document.activeElement as HTMLElement;

        // Shift + Tab: move backwards
        if (event.shiftKey) {
          if (activeElement === firstElement || !containerRef.current.contains(activeElement)) {
            event.preventDefault();
            lastElement.focus();
          }
        } else {
          // Tab: move forwards
          if (activeElement === lastElement || !containerRef.current.contains(activeElement)) {
            event.preventDefault();
            firstElement.focus();
          }
        }
      }
    },
    [isActive, containerRef, getFocusableElements, onEscape]
  );

  // Set up focus trap when activated
  useEffect(() => {
    if (isActive) {
      // Store currently focused element for restoration
      previousActiveElement.current = document.activeElement as HTMLElement;

      // Focus the initial element or first focusable element
      requestAnimationFrame(() => {
        if (initialFocusRef?.current) {
          initialFocusRef.current.focus();
        } else {
          const focusableElements = getFocusableElements();
          if (focusableElements.length > 0) {
            focusableElements[0].focus();
          } else {
            // If no focusable elements, focus the container itself
            containerRef.current?.focus();
          }
        }
      });

      // Add keyboard listener
      document.addEventListener("keydown", handleKeyDown);

      // Prevent body scroll when trap is active
      document.body.style.overflow = "hidden";

      return () => {
        document.removeEventListener("keydown", handleKeyDown);
        document.body.style.overflow = "";

        // Restore focus to previous element
        if (restoreFocus && previousActiveElement.current) {
          requestAnimationFrame(() => {
            previousActiveElement.current?.focus();
          });
        }
      };
    }
  }, [isActive, handleKeyDown, getFocusableElements, containerRef, restoreFocus, initialFocusRef]);

}
