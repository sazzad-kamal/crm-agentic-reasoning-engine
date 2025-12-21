import { memo } from "react";

/**
 * Skeleton loading component for message content.
 * Provides visual feedback while data is loading.
 */
export const MessageSkeleton = memo(function MessageSkeleton() {
  return (
    <div
      className="message-skeleton"
      role="status"
      aria-label="Loading message"
      aria-busy="true"
    >
      <div className="message-skeleton__line message-skeleton__line--short" />
      <div className="message-skeleton__line message-skeleton__line--long" />
      <div className="message-skeleton__line message-skeleton__line--medium" />
    </div>
  );
});

/**
 * Skeleton loading for the chat area.
 * Shows placeholder content while initial data loads.
 */
export const ChatSkeleton = memo(function ChatSkeleton() {
  return (
    <div
      className="chat-skeleton"
      role="status"
      aria-label="Loading chat"
      aria-busy="true"
    >
      <MessageSkeleton />
      <MessageSkeleton />
    </div>
  );
});
