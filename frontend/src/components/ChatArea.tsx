import { forwardRef } from "react";
import type { ChatMessage } from "../types";
import { MessageBlock } from "./MessageBlock";
import { EXAMPLE_PROMPTS } from "../config";

interface ChatAreaProps {
  messages: ChatMessage[];
  onSuggestionClick: (prompt: string) => void;
  onFollowUpClick: (question: string) => void;
}

/**
 * Main chat area showing messages or empty state
 */
export const ChatArea = forwardRef<HTMLDivElement, ChatAreaProps>(
  function ChatArea({ messages, onSuggestionClick, onFollowUpClick }, ref) {
    const isEmpty = messages.length === 0;

    return (
      <div className="chat-area" ref={ref}>
        {isEmpty ? (
          <EmptyState onSuggestionClick={onSuggestionClick} />
        ) : (
          <div className="message-list">
            {messages.map((msg) => (
              <MessageBlock
                key={msg.id}
                message={msg}
                onFollowUpClick={onFollowUpClick}
              />
            ))}
          </div>
        )}
      </div>
    );
  }
);

interface EmptyStateProps {
  onSuggestionClick: (prompt: string) => void;
}

/**
 * Empty state with example prompts
 */
function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <div className="empty-state__title">Try asking one of these questions:</div>
      <div className="empty-state__suggestions">
        {EXAMPLE_PROMPTS.map((prompt, index) => (
          <button
            key={index}
            className="suggestion-btn"
            onClick={() => onSuggestionClick(prompt)}
            type="button"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
