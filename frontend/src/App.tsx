import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { useChat } from "./hooks/useChat";
import {
  ChatArea,
  InputBar,
  ErrorBanner,
  ErrorBoundary,
  SkipLink,
} from "./components";
import "./styles/index.css";

/**
 * Acme CRM AI Companion - Main Application
 *
 * A conversational interface for querying CRM data using natural language.
 * Features:
 * - Natural language questions about accounts, activities, pipeline
 * - Real-time response with step-by-step progress
 * - Source citations and data tables
 * - Follow-up question suggestions
 * - Full keyboard accessibility with skip links
 * - ARIA live regions for screen reader support
 */
export default function App() {
  const [currentQuestion, setCurrentQuestion] = useState("");
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Memoized error handler to prevent hook re-initialization
  const chatOptions = useMemo(
    () => ({
      onError: (err: Error) => console.error("Chat error:", err),
    }),
    []
  );

  const { messages, isLoading, error, sendMessage, clearError } = useChat(chatOptions);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  // Focus input after sending a message
  useEffect(() => {
    if (!isLoading && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isLoading]);

  // Handle submitting a question
  const handleSubmit = useCallback(() => {
    if (currentQuestion.trim() && !isLoading) {
      sendMessage(currentQuestion);
      setCurrentQuestion("");
    }
  }, [currentQuestion, isLoading, sendMessage]);

  // Handle clicking an example prompt
  const handleSuggestionClick = useCallback(
    (prompt: string) => {
      if (!isLoading) {
        sendMessage(prompt);
      }
    },
    [isLoading, sendMessage]
  );

  // Handle clicking a follow-up suggestion - fills input for review
  const handleFollowUpClick = useCallback((question: string) => {
    setCurrentQuestion(question);
    // Focus input so user can review/edit before sending
    inputRef.current?.focus();
  }, []);

  // Handle onChange with stable reference
  const handleInputChange = useCallback((value: string) => {
    setCurrentQuestion(value);
  }, []);

  return (
    <ErrorBoundary>
      <SkipLink targetId="main-content" />
      <div className="page">
        <div className="container" role="main" id="main-content" tabIndex={-1}>
          {/* Header */}
          <header className="header">
            <h1 className="header__title">Acme CRM AI Companion</h1>
            <p className="header__subtitle">
              Ask questions about your CRM accounts, activity, and pipeline.
            </p>
          </header>

          {/* Chat Area */}
          <ChatArea
            ref={chatAreaRef}
            messages={messages}
            onSuggestionClick={handleSuggestionClick}
            onFollowUpClick={handleFollowUpClick}
          />

          {/* Error Banner */}
          {error && <ErrorBanner message={error} onDismiss={clearError} />}

          {/* Input Bar */}
          <InputBar
            ref={inputRef}
            value={currentQuestion}
            onChange={handleInputChange}
            onSubmit={handleSubmit}
            isLoading={isLoading}
          />
        </div>
      </div>
    </ErrorBoundary>
  );
}
