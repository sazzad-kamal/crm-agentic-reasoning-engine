import { memo } from "react";
import type { ChatMessage } from "../types";
import { config } from "../config";
import { LoadingState } from "./LoadingDots";
import { SourcesRow } from "./SourceChip";
import { StepsRow } from "./StepPill";
import { MetaInfo } from "./MetaInfo";
import { DataTables } from "./DataTables";
import { FollowUpSuggestions } from "./FollowUpSuggestions";

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
      <div className="message__label message__label--user" aria-hidden="true">
        You
      </div>
      <div className="message__question" role="heading" aria-level={3}>
        {message.question}
      </div>

      {/* Assistant Response */}
      <div className="message__label message__label--assistant" aria-hidden="true">
        Assistant
      </div>

      {response ? (
        <div className="message__response">
          {/* Answer Text */}
          <div className="message__answer">{response.answer}</div>

          {/* Sources */}
          {config.features.showSources && response.sources && response.sources.length > 0 && (
            <SourcesRow sources={response.sources} />
          )}

          {/* Meta Info */}
          {config.features.showLatency && response.meta && (
            <MetaInfo meta={response.meta} />
          )}

          {/* Steps */}
          {config.features.showSteps && response.steps && response.steps.length > 0 && (
            <StepsRow steps={response.steps} />
          )}

          {/* Data Tables */}
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
        <LoadingState />
      )}
    </article>
  );
});
