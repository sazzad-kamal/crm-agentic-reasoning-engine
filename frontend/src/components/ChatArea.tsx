import type { Ref } from "react";
import { useId, useState, useEffect } from "react";
import type { ChatMessage } from "../types";
import { MessageBlock } from "./MessageBlock";
import { EXAMPLE_PROMPTS, endpoints } from "../config";

interface ChatAreaProps {
  messages: ChatMessage[];
  onSuggestionClick: (prompt: string) => void;
  onFollowUpClick: (question: string) => void;
  /** App mode: "csv" (default) or "act" (demo mode) */
  mode?: "csv" | "act";
  /** Ref for scroll management (React 19 - ref as prop) */
  ref?: Ref<HTMLDivElement>;
}

/**
 * Main chat area showing messages or empty state.
 * React 19: Uses ref as a regular prop instead of forwardRef.
 */
export function ChatArea({
  messages,
  onSuggestionClick,
  onFollowUpClick,
  mode = "csv",
  ref,
}: ChatAreaProps) {
  const isEmpty = messages.length === 0;

  return (
    <div
      className="chat-area"
      ref={ref}
      role="log"
      aria-live="polite"
      aria-label="Chat messages"
    >
      {isEmpty ? (
        <EmptyState onSuggestionClick={onSuggestionClick} mode={mode} />
      ) : (
        <div className="message-list" role="list">
          {messages.map((msg, index) => {
            const isLastMessage = index === messages.length - 1;
            return (
              <MessageBlock
                key={msg.id}
                message={msg}
                onFollowUpClick={isLastMessage ? onFollowUpClick : undefined}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

interface EmptyStateProps {
  onSuggestionClick: (prompt: string) => void;
  mode: "csv" | "act";
}

/**
 * Empty state with illustration and example prompts.
 * Uses useId for unique, accessible label IDs.
 * Fetches dynamic starter questions from the question tree.
 */
function EmptyState({ onSuggestionClick, mode }: EmptyStateProps) {
  const suggestionsLabelId = useId();
  const [starterQuestions, setStarterQuestions] = useState<string[]>([...EXAMPLE_PROMPTS]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Fetch starter questions from question tree
    const fetchStarterQuestions = async () => {
      try {
        const response = await fetch(endpoints.starterQuestions);
        if (response.ok) {
          const data = await response.json();
          if (Array.isArray(data) && data.length > 0) {
            setStarterQuestions(data);
          }
        }
      } catch {
        // Fallback to static prompts on error (already set as default)
      } finally {
        setIsLoading(false);
      }
    };

    fetchStarterQuestions();
  }, []);

  return (
    <div className="empty-state" role="region" aria-label="Getting started">
      {/* Illustration */}
      <div className="empty-state__illustration">
        <svg viewBox="0 0 200 160" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Chat bubbles illustration */}
          <rect x="20" y="40" width="100" height="60" rx="12" fill="#EEF2FF" stroke="#C7D2FE" strokeWidth="2"/>
          <circle cx="45" cy="70" r="6" fill="#A5B4FC"/>
          <rect x="60" y="62" width="45" height="6" rx="3" fill="#C7D2FE"/>
          <rect x="60" y="74" width="35" height="6" rx="3" fill="#C7D2FE"/>
          
          <rect x="80" y="80" width="100" height="60" rx="12" fill="#F0FDF4" stroke="#86EFAC" strokeWidth="2"/>
          <circle cx="105" cy="110" r="6" fill="#10B981"/>
          <rect x="120" y="102" width="45" height="6" rx="3" fill="#86EFAC"/>
          <rect x="120" y="114" width="35" height="6" rx="3" fill="#86EFAC"/>
          
          {/* Sparkles */}
          <path d="M160 30L162 35L167 37L162 39L160 44L158 39L153 37L158 35L160 30Z" fill="#FBBF24"/>
          <path d="M40 20L41.5 24L45.5 25.5L41.5 27L40 31L38.5 27L34.5 25.5L38.5 24L40 20Z" fill="#6366F1"/>
          <path d="M175 70L176 73L179 74L176 75L175 78L174 75L171 74L174 73L175 70Z" fill="#10B981"/>
        </svg>
      </div>

      <h2 className="empty-state__heading">Welcome to Acme AI Companion</h2>

      <p className="empty-state__description">
        Click a question to explore your Act! CRM data
      </p>

      <div className="empty-state__title" id={suggestionsLabelId}>
        {mode === "act" ? "Select a question:" : "Try one of these to get started:"}
      </div>

      <div
        className="empty-state__suggestions"
        role="group"
        aria-labelledby={suggestionsLabelId}
      >
        {isLoading ? (
          // Show placeholder while loading
          <div className="empty-state__loading">Loading suggestions...</div>
        ) : (
          starterQuestions.map((prompt, index) => (
            <button
              key={index}
              className="suggestion-btn"
              onClick={() => onSuggestionClick(prompt)}
              type="button"
              aria-label={`Ask: ${prompt}`}
            >
              <span className="suggestion-btn__text">{prompt}</span>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
