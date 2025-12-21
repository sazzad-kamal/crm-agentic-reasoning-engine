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
 * Renders a single chat message with question and response
 */
export function MessageBlock({ message, onFollowUpClick }: MessageBlockProps) {
  const { response } = message;

  return (
    <div className="message-block">
      {/* User Question */}
      <div className="message__label message__label--user">You</div>
      <div className="message__question">{message.question}</div>

      {/* Assistant Response */}
      <div className="message__label message__label--assistant">Assistant</div>

      {response ? (
        <>
          {/* Answer Text */}
          <div className="message__answer">{response.answer}</div>

          {/* Sources */}
          {config.features.showSources && response.sources && (
            <SourcesRow sources={response.sources} />
          )}

          {/* Meta Info */}
          {config.features.showLatency && response.meta && (
            <MetaInfo meta={response.meta} />
          )}

          {/* Steps */}
          {config.features.showSteps && response.steps && (
            <StepsRow steps={response.steps} />
          )}

          {/* Data Tables */}
          {config.features.showDataTables && response.raw_data && (
            <DataTables rawData={response.raw_data} />
          )}

          {/* Follow-up Suggestions */}
          {config.features.showFollowUpSuggestions &&
            response.follow_up_suggestions &&
            onFollowUpClick && (
              <FollowUpSuggestions
                suggestions={response.follow_up_suggestions}
                onSuggestionClick={onFollowUpClick}
              />
            )}
        </>
      ) : (
        <LoadingState />
      )}
    </div>
  );
}
