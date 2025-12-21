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
      <div
        className="chat-area"
        ref={ref}
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        {isEmpty ? (
          <EmptyState onSuggestionClick={onSuggestionClick} />
        ) : (
          <div className="message-list" role="list">
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
 * Empty state with illustration and example prompts
 */
function EmptyState({ onSuggestionClick }: EmptyStateProps) {
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

      <h2 className="empty-state__heading">
        Welcome to Acme CRM AI
      </h2>
      
      <p className="empty-state__description">
        I can help you find information about your accounts, activities, pipeline, and more.
        Ask me anything in natural language!
      </p>

      <div className="empty-state__title" id="suggestions-label">
        Try one of these to get started:
      </div>
      
      <div
        className="empty-state__suggestions"
        role="group"
        aria-labelledby="suggestions-label"
      >
        {EXAMPLE_PROMPTS.map((prompt, index) => (
          <button
            key={index}
            className="suggestion-btn"
            onClick={() => onSuggestionClick(prompt)}
            type="button"
            aria-label={`Ask: ${prompt}`}
          >
            <span className="suggestion-btn__icon">💬</span>
            <span className="suggestion-btn__text">{prompt}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
