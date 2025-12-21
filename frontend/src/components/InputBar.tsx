import type { FormEvent, KeyboardEvent } from "react";
import { LoadingDots } from "./LoadingDots";

interface InputBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  placeholder?: string;
}

/**
 * Chat input bar with submit button
 */
export function InputBar({
  value,
  onChange,
  onSubmit,
  isLoading,
  placeholder = "Ask a question about your CRM...",
}: InputBarProps) {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!isLoading && value.trim()) {
      onSubmit();
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !isLoading && value.trim()) {
      e.preventDefault();
      onSubmit();
    }
  };

  const isDisabled = isLoading || !value.trim();

  return (
    <form className="input-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        className="input-bar__input"
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
        autoFocus
      />
      <button
        type="submit"
        className="input-bar__button"
        disabled={isDisabled}
      >
        {isLoading ? (
          <>
            <LoadingDots />
            <span>Thinking…</span>
          </>
        ) : (
          "Ask"
        )}
      </button>
    </form>
  );
}
