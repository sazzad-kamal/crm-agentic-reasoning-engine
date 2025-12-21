import { useState, useCallback, useRef, useEffect } from "react";
import type { ChatMessage, ChatResponse, ChatRequest } from "../types";
import { endpoints } from "../config";

interface UseChatOptions {
  onError?: (error: Error) => void;
}

interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (question: string, options?: Partial<ChatRequest>) => Promise<void>;
  clearMessages: () => void;
  clearError: () => void;
}

/**
 * Custom hook for managing chat state and API communication
 */
export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const sendMessage = useCallback(
    async (question: string, requestOptions?: Partial<ChatRequest>) => {
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion || isLoading) return;

      // Abort any pending request
      abortControllerRef.current?.abort();
      abortControllerRef.current = new AbortController();

      setError(null);
      setIsLoading(true);

      const messageId = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
      const newMessage: ChatMessage = {
        id: messageId,
        question: trimmedQuestion,
        response: null,
        timestamp: new Date(),
      };

      // Optimistically add the message
      setMessages((prev) => [...prev, newMessage]);

      try {
        const requestBody: ChatRequest = {
          question: trimmedQuestion,
          mode: requestOptions?.mode || "auto",
          company_id: requestOptions?.company_id,
          session_id: requestOptions?.session_id,
          user_id: requestOptions?.user_id,
        };

        const response = await fetch(endpoints.chat, {
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

        const data: ChatResponse = await response.json();

        // Update the message with the response
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId ? { ...msg, response: data } : msg
          )
        );
      } catch (err) {
        // Don't show error if request was aborted
        if (err instanceof Error && err.name === "AbortError") {
          return;
        }

        const errorMessage =
          err instanceof Error
            ? err.message
            : "Unable to reach the assistant. Please check that the backend is running.";

        console.error("Chat API error:", err);
        setError(errorMessage);
        options.onError?.(err instanceof Error ? err : new Error(errorMessage));

        // Remove the pending message on error
        setMessages((prev) => prev.filter((msg) => msg.id !== messageId));
      } finally {
        setIsLoading(false);
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
    error,
    sendMessage,
    clearMessages,
    clearError,
  };
}
