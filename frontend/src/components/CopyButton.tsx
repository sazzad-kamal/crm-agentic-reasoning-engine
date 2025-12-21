import { memo, useState, useCallback } from "react";

interface CopyButtonProps {
  text: string;
  className?: string;
}

/**
 * Copy to clipboard button with feedback
 */
export const CopyButton = memo(function CopyButton({ text, className = "" }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  }, [text]);

  return (
    <button
      className={`copy-button ${copied ? "copy-button--copied" : ""} ${className}`}
      onClick={handleCopy}
      type="button"
      aria-label={copied ? "Copied!" : "Copy to clipboard"}
      title={copied ? "Copied!" : "Copy to clipboard"}
    >
      {copied ? (
        <svg viewBox="0 0 24 24" fill="currentColor" className="copy-button__icon">
          <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="currentColor" className="copy-button__icon">
          <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z" />
        </svg>
      )}
    </button>
  );
});
