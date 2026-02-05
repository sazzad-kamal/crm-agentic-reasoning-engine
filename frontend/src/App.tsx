import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { useChatStream } from "./hooks/useChatStream";
import { useFocusTrap } from "./hooks/useFocusTrap";
import {
  ChatArea,
  InputBar,
  ErrorBanner,
  ErrorBoundary,
  SkipLink,
  DataExplorer,
} from "./components";
import { endpoints } from "./config";
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
 * - Data drawer to browse underlying CRM data
 * - Full keyboard accessibility with skip links
 * - ARIA live regions for screen reader support
 */
export default function App() {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const drawerRef = useRef<HTMLElement>(null);
  const drawerCloseRef = useRef<HTMLButtonElement>(null);

  // Demo mode state
  const [appInfo, setAppInfo] = useState<{ mode: "csv" | "act" } | null>(null);

  // Memoized error handler to prevent hook re-initialization
  const chatOptions = useMemo(
    () => ({
      onError: (err: Error) => console.error("Chat error:", err),
    }),
    []
  );

  // Chat hook with streaming
  const { messages, isLoading, error, sendMessage, clearError } = useChatStream(chatOptions);

  // Is demo mode active?
  const isDemoMode = appInfo?.mode === "act";

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  // Focus input after sending a message (only in non-demo mode)
  useEffect(() => {
    if (!isLoading && inputRef.current && !isDemoMode) {
      inputRef.current.focus();
    }
  }, [isLoading, isDemoMode]);

  // Fetch app info on mount
  useEffect(() => {
    fetch(endpoints.info)
      .then((r) => r.json())
      .then((data) => setAppInfo({ mode: data.mode }))
      .catch(() => setAppInfo({ mode: "csv" }));
  }, []);

  // Close drawer handler (stable reference for focus trap)
  const closeDrawer = useCallback(() => {
    setIsDrawerOpen(false);
  }, []);

  // Focus trap for drawer accessibility (handles Escape and Tab cycling)
  useFocusTrap(drawerRef, {
    isActive: isDrawerOpen,
    onEscape: closeDrawer,
    restoreFocus: true,
    initialFocusRef: drawerCloseRef,
  });

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

  // Handle clicking a follow-up suggestion - auto-sends like starter questions
  const handleFollowUpClick = useCallback(
    (question: string) => {
      if (!isLoading) {
        sendMessage(question);
      }
    },
    [isLoading, sendMessage]
  );

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

  // Update document title based on state
  useEffect(() => {
    const appName = "Acme AI Companion";
    const pageTitle = isLoading
      ? `Thinking... | ${appName}`
      : messages.length > 0
        ? `${messages.length} messages | ${appName}`
        : appName;

    document.title = pageTitle;
  }, [isLoading, messages.length, isDemoMode]);

  return (
    <ErrorBoundary>
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
              <h1 className="header__title">Acme AI Companion</h1>
              <p className="header__subtitle">
                AI-powered insights for your Act! CRM
              </p>
            </div>

            {/* Browse Data button - hidden in demo mode */}
            {!isDemoMode && (
              <button
                className="header__data-btn"
                onClick={() => setIsDrawerOpen(true)}
                aria-label="Browse CRM data"
                title="Browse the data the AI has access to"
              >
                <span className="header__data-btn-icon">📊</span>
                <span className="header__data-btn-text">Browse Data</span>
              </button>
            )}
          </header>

          {/* Chat Area */}
          <ChatArea
            ref={chatAreaRef}
            messages={messages}
            onSuggestionClick={handleSuggestionClick}
            onFollowUpClick={handleFollowUpClick}
            mode={appInfo?.mode}
          />

          {/* Error Banner */}
          {error && <ErrorBanner message={error} onDismiss={clearError} />}

          {/* Input Bar - hidden in demo mode */}
          {!isDemoMode && (
            <InputBar
              ref={inputRef}
              value={currentQuestion}
              onChange={handleInputChange}
              onSubmit={handleSubmit}
              isLoading={isLoading}
            />
          )}
        </div>

        {/* Data Drawer Overlay - only in non-demo mode */}
        {!isDemoMode && isDrawerOpen && (
          <div
            className="drawer-overlay"
            onClick={closeDrawer}
            aria-hidden="true"
          />
        )}

        {/* Data Drawer - only in non-demo mode */}
        {!isDemoMode && (
          <aside
            ref={drawerRef}
            className={`drawer ${isDrawerOpen ? "drawer--open" : ""}`}
            role="dialog"
            aria-modal="true"
            aria-label="CRM Data Browser"
            tabIndex={-1}
          >
            <div className="drawer__header">
              <h2 className="drawer__title">CRM Data</h2>
              <button
                ref={drawerCloseRef}
                className="drawer__close"
                onClick={closeDrawer}
                aria-label="Close data browser"
              >
                ✕
              </button>
            </div>
            <div className="drawer__content">
              <DataExplorer onAskAbout={handleAskAbout} />
            </div>
          </aside>
        )}
      </div>
    </ErrorBoundary>
  );
}
