import { useState, useCallback, useRef, useEffect } from "react";
import type { ChatMessage, ChatResponse, ChatRequest, SectionStatus, RawData } from "../types";
import { config } from "../config";

// =============================================================================
// HTTP Helpers (inlined - single use)
// =============================================================================

const CONNECTION_ERROR_MESSAGE =
  "Unable to reach the assistant. Please check that the backend is running.";

async function checkHttpResponse(response: Response): Promise<void> {
  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error");
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === "AbortError";
}

// =============================================================================
// Types
// =============================================================================

interface StreamEvent {
  type: string;
  data: Record<string, unknown>;
}

interface UseChatStreamOptions {
  onError?: (error: Error) => void;
}

interface UseChatStreamReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;
  sendMessage: (question: string) => Promise<void>;
  clearMessages: () => void;
  clearError: () => void;
  abortStream: () => void;
}

// =============================================================================
// Stream Event Parser
// =============================================================================

function parseSSE(text: string): StreamEvent[] {
  const events: StreamEvent[] = [];
  const lines = text.split("\n");

  let currentEvent = "";
  let currentData = "";

  for (const line of lines) {
    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      currentData = line.slice(6);
    } else if (line === "" && currentEvent && currentData) {
      try {
        events.push({
          type: currentEvent,
          data: JSON.parse(currentData),
        });
      } catch {
        console.warn("Failed to parse SSE data:", currentData);
      }
      currentEvent = "";
      currentData = "";
    }
  }

  return events;
}

// =============================================================================
// Hook
// =============================================================================

// Generate a unique session ID (persists for the browser tab lifetime)
function generateSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Custom hook for streaming chat with real-time updates
 */
export function useChatStream(options: UseChatStreamOptions = {}): UseChatStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);

  // Session ID persists across the hook's lifetime (one per browser tab)
  const sessionIdRef = useRef<string>(generateSessionId());

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      readerRef.current?.cancel();
    };
  }, []);

  const abortStream = useCallback(() => {
    abortControllerRef.current?.abort();
    readerRef.current?.cancel();
    setIsStreaming(false);
    setIsLoading(false);
  }, []);

  const sendMessage = useCallback(
    async (question: string) => {
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion || isLoading) return;

      // Abort any pending request
      abortControllerRef.current?.abort();
      readerRef.current?.cancel();
      abortControllerRef.current = new AbortController();

      setError(null);
      setIsLoading(true);
      setIsStreaming(true);

      const messageId = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

      // Initialize message with all sections loading
      const sectionStatus: SectionStatus = {
        data: "loading",
        answer: "loading",
        action: "loading",
        followup: "loading",
      };

      const newMessage: ChatMessage = {
        id: messageId,
        question: trimmedQuestion,
        response: { answer: "" },
        sectionStatus: { ...sectionStatus },
        timestamp: new Date(),
      };

      // Optimistically add the message
      setMessages((prev) => [...prev, newMessage]);

      // Build up the response as events arrive
      let accumulatedAnswer = "";
      let accumulatedSqlResults: RawData | undefined;
      let accumulatedAction: string | null | undefined;
      let accumulatedFollowUps: string[] | undefined;
      let finalResponse: ChatResponse | null = null;

      // Helper to update message with current state
      const updateMessage = () => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? {
                  ...msg,
                  response: {
                    answer: accumulatedAnswer,
                    ...(accumulatedSqlResults !== undefined && { sql_results: accumulatedSqlResults }),
                    ...(accumulatedAction !== undefined && { suggested_action: accumulatedAction }),
                    ...(accumulatedFollowUps !== undefined && { follow_up_suggestions: accumulatedFollowUps }),
                  },
                  sectionStatus: { ...sectionStatus },
                }
              : msg
          )
        );
      };

      try {
        const requestBody: ChatRequest = {
          question: trimmedQuestion,
          session_id: sessionIdRef.current,
        };

        const response = await fetch(`${config.apiUrl}/api/chat/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current.signal,
        });

        await checkHttpResponse(response);

        // Read the stream
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No response body");
        }
        readerRef.current = reader;

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete events in buffer
          const events = parseSSE(buffer);

          // Keep incomplete data in buffer
          const lastNewline = buffer.lastIndexOf("\n\n");
          if (lastNewline !== -1) {
            buffer = buffer.slice(lastNewline + 2);
          }

          for (const event of events) {
            switch (event.type) {
              case "data_ready": {
                const rawData = event.data.sql_results as RawData | undefined;
                // Only set sql_results if it contains actual data (hide empty data section)
                accumulatedSqlResults = rawData && Object.keys(rawData).length > 0 ? rawData : undefined;
                sectionStatus.data = "done";
                updateMessage();
                break;
              }

              case "answer_chunk":
                accumulatedAnswer += event.data.chunk as string;
                sectionStatus.answer = "done";
                updateMessage();
                break;

              case "action_chunk":
                accumulatedAction = (accumulatedAction ?? "") + (event.data.chunk as string);
                sectionStatus.action = "done";
                updateMessage();
                break;

              case "action_ready":
                accumulatedAction = (event.data.suggested_action as string | null) ?? null;
                sectionStatus.action = "done";
                updateMessage();
                break;

              case "followup_ready":
                accumulatedFollowUps = (event.data.follow_up_suggestions as string[]) ?? [];
                sectionStatus.followup = "done";
                updateMessage();
                break;

              case "done":
                accumulatedAnswer = event.data.answer as string;
                finalResponse = event.data as unknown as ChatResponse;
                break;

              case "error":
                throw new Error(event.data.message as string);

              default:
                break;
            }
          }
        }

        // Apply final response and remove sectionStatus (streaming complete).
        // Accumulated section data (from per-section events) takes priority over
        // the done event's payload, which may contain empty defaults like sql_results: {}.
        if (finalResponse) {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === messageId
                ? {
                    ...msg,
                    response: {
                      ...finalResponse,
                      answer: accumulatedAnswer,
                      ...(accumulatedSqlResults !== undefined && { sql_results: accumulatedSqlResults }),
                      ...(accumulatedAction !== undefined && { suggested_action: accumulatedAction }),
                      ...(accumulatedFollowUps !== undefined && { follow_up_suggestions: accumulatedFollowUps }),
                    },
                    sectionStatus: undefined,
                  }
                : msg
            )
          );
        }

      } catch (err) {
        if (isAbortError(err)) return;

        const errorMessage =
          err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE;

        console.error("Chat stream error:", err);
        setError(errorMessage);
        options.onError?.(err instanceof Error ? err : new Error(errorMessage));

        // Remove the pending message on error
        setMessages((prev) => prev.filter((msg) => msg.id !== messageId));
      } finally {
        setIsLoading(false);
        setIsStreaming(false);
        readerRef.current = null;
      }
    },
    [isLoading, options]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    isStreaming,
    error,
    sendMessage,
    clearMessages,
    clearError,
    abortStream,
  };
}
