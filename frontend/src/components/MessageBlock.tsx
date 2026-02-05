import { memo } from "react";
import type { ChatMessage } from "../types";
import { config } from "../config";
import { DataTables } from "./DataTables";
import { FollowUpSuggestions } from "./FollowUpSuggestions";
import { ProgressChecklist } from "./ProgressChecklist";
import { SuggestedActions } from "./SuggestedActions";
import { Avatar } from "./Avatar";
import { CopyButton } from "./CopyButton";
import { MarkdownText } from "./MarkdownText";

/** Skeleton: 3 text lines of varying width (answer placeholder). */
function AnswerSkeleton() {
  return (
    <div className="skeleton-answer" role="status" aria-label="Generating answer">
      <div className="skeleton-answer__line skeleton-answer__line--long" />
      <div className="skeleton-answer__line skeleton-answer__line--medium" />
      <div className="skeleton-answer__line skeleton-answer__line--short" />
      <span className="visually-hidden">Generating answer...</span>
    </div>
  );
}

/** Skeleton: single rounded bar (action placeholder). */
function ActionSkeleton() {
  return (
    <div className="skeleton-action" role="status" aria-label="Loading suggested action">
      <div className="skeleton-action__bar" />
    </div>
  );
}

/** Skeleton: small table grid (data placeholder). */
function DataSkeleton() {
  return (
    <div className="skeleton-data" role="status" aria-label="Loading data">
      <div className="skeleton-data__row">
        <div className="skeleton-data__cell skeleton-data__cell--header" />
        <div className="skeleton-data__cell skeleton-data__cell--header" />
        <div className="skeleton-data__cell skeleton-data__cell--header" />
      </div>
      <div className="skeleton-data__row">
        <div className="skeleton-data__cell" />
        <div className="skeleton-data__cell" />
        <div className="skeleton-data__cell" />
      </div>
      <div className="skeleton-data__row">
        <div className="skeleton-data__cell" />
        <div className="skeleton-data__cell" />
        <div className="skeleton-data__cell" />
      </div>
    </div>
  );
}

/** Skeleton: inline pill chips (follow-up placeholder). */
function FollowUpSkeleton() {
  return (
    <div className="skeleton-followup" role="status" aria-label="Loading suggestions">
      <div className="skeleton-followup__pill" />
      <div className="skeleton-followup__pill" />
      <div className="skeleton-followup__pill" />
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
  const { response, sectionStatus } = message;
  const isStreaming = sectionStatus !== undefined;

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

          <div className="message__response">
            {/* Answer Section */}
            {isStreaming && sectionStatus.answer === "loading" ? (
              <AnswerSkeleton />
            ) : response?.answer ? (
              <div className="message__answer-wrapper">
                <MarkdownText text={response.answer} className="message__answer" />
                {!isStreaming && <CopyButton text={response.answer} className="message__copy" />}
              </div>
            ) : null}

            {/* Suggested Actions Section */}
            {isStreaming && sectionStatus.action === "loading" ? (
              <ActionSkeleton />
            ) : response?.suggested_action ? (
              <SuggestedActions action={response.suggested_action} />
            ) : null}

            {/* Data Tables Section */}
            {isStreaming && sectionStatus.data === "loading" ? (
              // Show progress checklist if steps available, otherwise show skeleton
              response?.fetchSteps && response.fetchSteps.length > 0 ? (
                <ProgressChecklist steps={response.fetchSteps} />
              ) : (
                <DataSkeleton />
              )
            ) : config.features.showDataTables && response?.sql_results ? (
              <DataTables rawData={response.sql_results} />
            ) : null}

            {/* Follow-up Suggestions Section */}
            {isStreaming && sectionStatus.followup === "loading" ? (
              <FollowUpSkeleton />
            ) : config.features.showFollowUpSuggestions &&
              response?.follow_up_suggestions &&
              response.follow_up_suggestions.length > 0 &&
              onFollowUpClick ? (
              <FollowUpSuggestions
                suggestions={response.follow_up_suggestions}
                onSuggestionClick={onFollowUpClick}
              />
            ) : null}
          </div>
        </div>
      </div>
    </article>
  );
});
