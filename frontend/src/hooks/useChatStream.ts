import { useState, useCallback, useRef, useEffect } from "react";
import type { ChatMessage, ChatResponse, ChatRequest, Source, Step } from "../types";
import { config } from "../config";

// =============================================================================
// Types
// =============================================================================

interface StreamEvent {
  type: string;
  data: Record<string, unknown>;
}

interface UseChatStreamOptions {
  onError?: (error: Error) => void;
  onStatusUpdate?: (message: string) => void;
}

interface UseChatStreamReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;
  currentStatus: string | null;
  sendMessage: (question: string, options?: Partial<ChatRequest>) => Promise<void>;
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

/**
 * Custom hook for streaming chat with real-time updates
 */
export function useChatStream(options: UseChatStreamOptions = {}): UseChatStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);

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
    setCurrentStatus(null);
  }, []);

  const sendMessage = useCallback(
    async (question: string, requestOptions?: Partial<ChatRequest>) => {
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion || isLoading) return;

      // Abort any pending request
      abortControllerRef.current?.abort();
      readerRef.current?.cancel();
      abortControllerRef.current = new AbortController();

      setError(null);
      setIsLoading(true);
      setIsStreaming(true);
      setCurrentStatus("Starting...");

      const messageId = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
      
      // Initialize message with empty response (will be built up)
      const newMessage: ChatMessage = {
        id: messageId,
        question: trimmedQuestion,
        response: null,
        timestamp: new Date(),
      };

      // Optimistically add the message
      setMessages((prev) => [...prev, newMessage]);

      // Build up the response as events arrive
      let accumulatedAnswer = "";
      let accumulatedSources: Source[] = [];
      const accumulatedSteps: Step[] = [];
      let finalResponse: ChatResponse | null = null;

      try {
        const requestBody: ChatRequest = {
          question: trimmedQuestion,
          mode: requestOptions?.mode || "auto",
          company_id: requestOptions?.company_id,
          session_id: requestOptions?.session_id,
          user_id: requestOptions?.user_id,
        };

        const response = await fetch(`${config.apiUrl}/api/chat/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          const errorText = await response.text().catch(() => "Unknown error");
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

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
              case "status":
                setCurrentStatus(event.data.message as string);
                options.onStatusUpdate?.(event.data.message as string);
                break;
                
              case "step":
                accumulatedSteps.push(event.data as unknown as Step);
                // Update message with current steps
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === messageId
                      ? {
                          ...msg,
                          response: {
                            answer: accumulatedAnswer,
                            sources: accumulatedSources,
                            steps: [...accumulatedSteps],
                          },
                        }
                      : msg
                  )
                );
                break;
                
              case "sources": {
                const newSources = (event.data.sources as Source[]) || [];
                accumulatedSources = [...accumulatedSources, ...newSources];
                break;
              }
                
              case "answer_start":
                setCurrentStatus("Generating answer...");
                break;
                
              case "answer_chunk":
                accumulatedAnswer += event.data.chunk as string;
                // Update message with current answer
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === messageId
                      ? {
                          ...msg,
                          response: {
                            answer: accumulatedAnswer,
                            sources: accumulatedSources,
                            steps: accumulatedSteps,
                          },
                        }
                      : msg
                  )
                );
                break;
                
              case "answer_end":
                accumulatedAnswer = event.data.answer as string;
                break;
                
              case "followup":
                // Will be included in final done event
                break;
                
              case "done":
                finalResponse = event.data as unknown as ChatResponse;
                setCurrentStatus(null);
                break;
                
              case "error":
                throw new Error(event.data.message as string);
            }
          }
        }

        // Apply final response
        if (finalResponse) {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === messageId ? { ...msg, response: finalResponse } : msg
            )
          );
        }

      } catch (err) {
        // Don't show error if request was aborted
        if (err instanceof Error && err.name === "AbortError") {
          return;
        }

        const errorMessage =
          err instanceof Error
            ? err.message
            : "Unable to reach the assistant. Please check that the backend is running.";

        console.error("Chat stream error:", err);
        setError(errorMessage);
        options.onError?.(err instanceof Error ? err : new Error(errorMessage));

        // Remove the pending message on error
        setMessages((prev) => prev.filter((msg) => msg.id !== messageId));
      } finally {
        setIsLoading(false);
        setIsStreaming(false);
        setCurrentStatus(null);
        readerRef.current = null;
      }
    },
    [isLoading, options]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    setCurrentStatus(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    isStreaming,
    error,
    currentStatus,
    sendMessage,
    clearMessages,
    clearError,
    abortStream,
  };
}
