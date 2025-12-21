import { useState, useRef, useEffect, useCallback } from "react";
import { useChat } from "./hooks/useChat";
import { ChatArea, InputBar, ErrorBanner } from "./components";
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
 */
export default function App() {
  const [currentQuestion, setCurrentQuestion] = useState("");
  const chatAreaRef = useRef<HTMLDivElement>(null);

  const { messages, isLoading, error, sendMessage, clearError } = useChat({
    onError: (err) => console.error("Chat error:", err),
  });

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

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

  // Handle clicking a follow-up suggestion
  const handleFollowUpClick = useCallback((question: string) => {
    setCurrentQuestion(question);
  }, []);

  return (
    <div className="page">
      <div className="container">
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
          value={currentQuestion}
          onChange={setCurrentQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}
