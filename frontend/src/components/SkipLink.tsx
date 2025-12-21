import { memo } from "react";

interface SkipLinkProps {
  /** ID of the main content element to skip to */
  targetId?: string;
  /** Custom text for the skip link */
  children?: string;
}

/**
 * Skip link component for keyboard accessibility.
 * Allows keyboard users to bypass navigation and jump to main content.
 * Visually hidden until focused.
 */
export const SkipLink = memo(function SkipLink({
  targetId = "main-content",
  children = "Skip to main content",
}: SkipLinkProps) {
  return (
    <a
      href={`#${targetId}`}
      className="skip-link"
      tabIndex={0}
    >
      {children}
    </a>
  );
});
