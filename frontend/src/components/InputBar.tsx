import { forwardRef, memo, useCallback } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { LoadingDots } from "./LoadingDots";

interface InputBarProps {
  /** Current input value */
  value: string;
  /** Callback when input value changes */
  onChange: (value: string) => void;
  /** Callback when form is submitted */
  onSubmit: () => void;
  /** Whether a request is in progress */
  isLoading: boolean;
  /** Placeholder text for the input */
  placeholder?: string;
}

/**
 * Chat input bar with submit button.
 * Supports forwarded refs for focus management.
 * Memoized to prevent unnecessary re-renders.
 */
export const InputBar = memo(
  forwardRef<HTMLInputElement, InputBarProps>(function InputBar(
    {
      value,
      onChange,
      onSubmit,
      isLoading,
      placeholder = "Ask a question about your CRM...",
    },
    ref
  ) {
    const handleSubmit = useCallback(
      (e: FormEvent) => {
        e.preventDefault();
        if (!isLoading && value.trim()) {
          onSubmit();
        }
      },
      [isLoading, value, onSubmit]
    );

    const handleKeyDown = useCallback(
      (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey && !isLoading && value.trim()) {
          e.preventDefault();
          onSubmit();
        }
      },
      [isLoading, value, onSubmit]
    );

    const handleChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        onChange(e.target.value);
      },
      [onChange]
    );

    const isDisabled = isLoading || !value.trim();

    return (
      <form
        className="input-bar"
        onSubmit={handleSubmit}
        role="search"
        aria-label="Ask a question"
      >
        <input
          ref={ref}
          type="text"
          className="input-bar__input"
          placeholder={placeholder}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          autoFocus
          aria-label="Ask a question about your CRM"
          aria-describedby="input-hint"
        />
        <span id="input-hint" className="visually-hidden">
          Press Enter to send your question
        </span>
        <button
          type="submit"
          className="input-bar__button"
          disabled={isDisabled}
          aria-label={isLoading ? "Sending message" : "Send message"}
        >
          {isLoading ? (
            <>
              <LoadingDots />
              <span>Thinking…</span>
            </>
          ) : (
            "Send"
          )}
        </button>
      </form>
    );
  })
);
