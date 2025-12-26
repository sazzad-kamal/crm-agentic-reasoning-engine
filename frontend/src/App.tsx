import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { useChat } from "./hooks/useChat";
import { useChatStream } from "./hooks/useChatStream";
import {
  ChatArea,
  InputBar,
  ErrorBanner,
  ErrorBoundary,
  SkipLink,
  DataExplorer,
} from "./components";
import "./styles/index.css";

// Feature flag for streaming - can be disabled if issues occur
const USE_STREAMING = true;

/**
 * Acme CRM AI Companion - Main Application
 *
 * A conversational interface for querying CRM data using natural language.
 * Features:
 * - Natural language questions about accounts, activities, pipeline
 * - Real-time response with step-by-step progress
 * - Source citations and data tables
 * - Follow-up question suggestions
 * - Data drawer to browse underlying CRM data
 * - Full keyboard accessibility with skip links
 * - ARIA live regions for screen reader support
 */
export default function App() {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
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

  // Use streaming or regular chat hook based on feature flag
  const streamingChat = useChatStream(chatOptions);
  const regularChat = useChat(chatOptions);
  
  // Select which hook to use
  const chat = USE_STREAMING ? streamingChat : regularChat;
  const { messages, isLoading, error, sendMessage, clearError } = chat;
  const currentStatus = USE_STREAMING && 'currentStatus' in chat ? (chat.currentStatus as string | null) : null;
  const isStreaming = USE_STREAMING && 'isStreaming' in chat ? (chat.isStreaming as boolean) : false;

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

  // Close drawer on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isDrawerOpen) {
        setIsDrawerOpen(false);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isDrawerOpen]);

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
    inputRef.current?.focus();
  }, []);

  // Handle onChange with stable reference
  const handleInputChange = useCallback((value: string) => {
    setCurrentQuestion(value);
  }, []);

  // Handle "Ask AI" from data explorer
  const handleAskAbout = useCallback((question: string) => {
    setCurrentQuestion(question);
    setIsDrawerOpen(false);
    setTimeout(() => inputRef.current?.focus(), 100);
  }, []);

  // Dynamic page title based on state (React 19 feature)
  const pageTitle = isLoading 
    ? "Thinking... | Acme CRM AI"
    : messages.length > 0 
      ? `${messages.length} messages | Acme CRM AI`
      : "Acme CRM AI Companion";

  return (
    <ErrorBoundary>
      {/* React 19: Document metadata rendered in components */}
      <title>{pageTitle}</title>
      <meta name="description" content="AI-powered assistant for querying your CRM data" />
      
      <SkipLink targetId="main-content" />
      <div className="page">
        <div className="container" role="main" id="main-content" tabIndex={-1}>
          {/* Header */}
          <header className="header">
            <div className="header__logo">
              <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" className="header__icon">
                <rect width="40" height="40" rx="10" fill="url(#gradient)"/>
                <path d="M12 20C12 15.5817 15.5817 12 20 12C24.4183 12 28 15.5817 28 20C28 24.4183 24.4183 28 20 28" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
                <path d="M20 28C17.7909 28 16 26.2091 16 24C16 21.7909 17.7909 20 20 20C22.2091 20 24 21.7909 24 24" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
                <circle cx="20" cy="24" r="2" fill="white"/>
                <defs>
                  <linearGradient id="gradient" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#6366F1"/>
                    <stop offset="1" stopColor="#8B5CF6"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div className="header__text">
              <h1 className="header__title">Acme CRM AI Companion</h1>
              <p className="header__subtitle">
                Ask questions about your CRM accounts, activity, and pipeline.
              </p>
            </div>
            <button
              className="header__data-btn"
              onClick={() => setIsDrawerOpen(true)}
              aria-label="Browse CRM data"
              title="Browse the data the AI has access to"
            >
              <span className="header__data-btn-icon">📊</span>
              <span className="header__data-btn-text">Browse Data</span>
            </button>
          </header>

          {/* Chat Area */}
          <ChatArea
            ref={chatAreaRef}
            messages={messages}
            onSuggestionClick={handleSuggestionClick}
            onFollowUpClick={handleFollowUpClick}
            streamingStatus={isStreaming ? currentStatus : null}
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

        {/* Data Drawer Overlay */}
        {isDrawerOpen && (
          <div 
            className="drawer-overlay" 
            onClick={() => setIsDrawerOpen(false)}
            aria-hidden="true"
          />
        )}

        {/* Data Drawer */}
        <aside
          className={`drawer ${isDrawerOpen ? "drawer--open" : ""}`}
          role="dialog"
          aria-modal="true"
          aria-label="CRM Data Browser"
        >
          <div className="drawer__header">
            <h2 className="drawer__title">CRM Data</h2>
            <button
              className="drawer__close"
              onClick={() => setIsDrawerOpen(false)}
              aria-label="Close data browser"
            >
              ✕
            </button>
          </div>
          <div className="drawer__content">
            <DataExplorer onAskAbout={handleAskAbout} />
          </div>
        </aside>
      </div>
    </ErrorBoundary>
  );
}
