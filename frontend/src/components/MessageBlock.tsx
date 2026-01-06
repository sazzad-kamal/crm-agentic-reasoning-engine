import { memo } from "react";
import type { ChatMessage } from "../types";
import { config } from "../config";
import { DataTables } from "./DataTables";
import { FollowUpSuggestions } from "./FollowUpSuggestions";
import { Avatar } from "./Avatar";
import { CopyButton } from "./CopyButton";
import { MarkdownText } from "./MarkdownText";

/**
 * Skeleton loader shown while waiting for response.
 * Shows message-shaped placeholder with shimmer animation.
 */
function MessageSkeleton() {
  return (
    <div className="message-skeleton" role="status" aria-label="Assistant is thinking">
      <div className="message-skeleton__line message-skeleton__line--long" />
      <div className="message-skeleton__line message-skeleton__line--medium" />
      <div className="message-skeleton__line message-skeleton__line--short" />
      <span className="visually-hidden">Loading response...</span>
    </div>
  );
}

interface MessageBlockProps {
  message: ChatMessage;
  onFollowUpClick?: (question: string) => void;
}

/**
 * Renders a single chat message with question and response.
 * Memoized to prevent unnecessary re-renders in the message list.
 */
export const MessageBlock = memo(function MessageBlock({
  message,
  onFollowUpClick,
}: MessageBlockProps) {
  const { response } = message;

  return (
    <article
      className="message-block"
      role="listitem"
      aria-label={`Conversation about: ${message.question.slice(0, 50)}${message.question.length > 50 ? "..." : ""}`}
    >
      {/* User Question */}
      <div className="message__row message__row--user">
        <Avatar type="user" />
        <div className="message__content">
          <div className="message__label message__label--user" aria-hidden="true">
            You
          </div>
          <div className="message__question" role="heading" aria-level={3}>
            {message.question}
          </div>
        </div>
      </div>

      {/* Assistant Response */}
      <div className="message__row message__row--assistant">
        <Avatar type="assistant" />
        <div className="message__content">
          <div className="message__label message__label--assistant" aria-hidden="true">
            Assistant
          </div>

          {response?.answer ? (
            <div className="message__response">
              {/* Answer Text with Copy Button */}
              <div className="message__answer-wrapper">
                <MarkdownText text={response.answer} className="message__answer" />
                <CopyButton text={response.answer} className="message__copy" />
              </div>

              {/* Data Tables (collapsed) */}
              {config.features.showDataTables && response.raw_data && (
                <DataTables rawData={response.raw_data} />
              )}

              {/* Follow-up Suggestions */}
              {config.features.showFollowUpSuggestions &&
                response.follow_up_suggestions &&
                response.follow_up_suggestions.length > 0 &&
                onFollowUpClick && (
                  <FollowUpSuggestions
                    suggestions={response.follow_up_suggestions}
                    onSuggestionClick={onFollowUpClick}
                  />
                )}
            </div>
          ) : (
            <MessageSkeleton />
          )}
        </div>
      </div>
    </article>
  );
});
