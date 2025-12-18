import { useState, useRef, useEffect } from "react";

// =============================================================================
// TypeScript Interfaces
// =============================================================================

interface Source {
  type: string;
  id: string;
  label: string;
}

interface Step {
  id: string;
  label: string;
  status: "done" | "pending" | "running";
}

interface Company {
  company_id: string;
  name: string;
  plan: string;
  renewal_date: string;
}

interface Activity {
  activity_id: string;
  type: string;
  occurred_at: string;
  owner: string;
  summary?: string;
  subject?: string;
}

interface Opportunity {
  opportunity_id: string;
  name: string;
  stage: string;
  expected_close_date: string;
  value: number;
}

interface RawData {
  companies?: Company[];
  activities?: Activity[];
  opportunities?: Opportunity[];
}

interface Meta {
  mode_used?: string;
  latency_ms?: number;
  model?: string;
}

interface ChatResponse {
  answer: string;
  sources?: Source[];
  steps?: Step[];
  raw_data?: RawData;
  meta?: Meta;
}

interface ChatMessage {
  id: string;
  question: string;
  response: ChatResponse | null;
}

// =============================================================================
// Styles
// =============================================================================

const styles: { [key: string]: React.CSSProperties } = {
  page: {
    minHeight: "100vh",
    backgroundColor: "#f5f7fa",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  container: {
    maxWidth: 960,
    margin: "0 auto",
    padding: "24px 16px",
  },
  header: {
    textAlign: "center",
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    color: "#1a1a2e",
    margin: 0,
  },
  subtitle: {
    fontSize: 14,
    color: "#6b7280",
    marginTop: 8,
  },
  chatArea: {
    backgroundColor: "#ffffff",
    borderRadius: 12,
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
    minHeight: 400,
    maxHeight: "calc(100vh - 320px)",
    overflowY: "auto" as const,
    marginBottom: 16,
  },
  emptyState: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    padding: 48,
    height: "100%",
    minHeight: 400,
  },
  emptyTitle: {
    fontSize: 18,
    color: "#374151",
    marginBottom: 24,
  },
  suggestionsContainer: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 12,
    width: "100%",
    maxWidth: 500,
  },
  suggestionButton: {
    padding: "12px 16px",
    backgroundColor: "#f3f4f6",
    border: "1px solid #e5e7eb",
    borderRadius: 8,
    cursor: "pointer",
    textAlign: "left" as const,
    fontSize: 14,
    color: "#374151",
    transition: "all 0.15s ease",
  },
  messageList: {
    padding: 16,
  },
  messageBlock: {
    marginBottom: 24,
    paddingBottom: 24,
    borderBottom: "1px solid #e5e7eb",
  },
  userLabel: {
    fontSize: 12,
    fontWeight: 600,
    color: "#6366f1",
    marginBottom: 4,
    textTransform: "uppercase" as const,
    letterSpacing: "0.5px",
  },
  userQuestion: {
    fontSize: 15,
    color: "#1f2937",
    marginBottom: 16,
  },
  assistantLabel: {
    fontSize: 12,
    fontWeight: 600,
    color: "#10b981",
    marginBottom: 4,
    textTransform: "uppercase" as const,
    letterSpacing: "0.5px",
  },
  answerText: {
    fontSize: 15,
    color: "#374151",
    lineHeight: 1.6,
    marginBottom: 16,
  },
  sourcesRow: {
    display: "flex",
    flexWrap: "wrap" as const,
    gap: 8,
    marginBottom: 12,
  },
  sourceChip: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "4px 10px",
    backgroundColor: "#eff6ff",
    color: "#1d4ed8",
    borderRadius: 16,
    fontSize: 12,
    fontWeight: 500,
  },
  sourceIcon: {
    fontSize: 10,
  },
  metaLine: {
    fontSize: 12,
    color: "#9ca3af",
    marginBottom: 12,
  },
  stepsRow: {
    display: "flex",
    flexWrap: "wrap" as const,
    gap: 8,
    marginBottom: 12,
  },
  stepPill: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "4px 10px",
    borderRadius: 16,
    fontSize: 12,
    fontWeight: 500,
  },
  stepDone: {
    backgroundColor: "#d1fae5",
    color: "#065f46",
  },
  stepPending: {
    backgroundColor: "#fef3c7",
    color: "#92400e",
  },
  stepRunning: {
    backgroundColor: "#dbeafe",
    color: "#1e40af",
  },
  collapsibleHeader: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 600,
    color: "#6b7280",
    marginTop: 12,
    marginBottom: 8,
  },
  dataSection: {
    backgroundColor: "#f9fafb",
    borderRadius: 8,
    padding: 16,
    marginTop: 8,
  },
  tableTitle: {
    fontSize: 12,
    fontWeight: 600,
    color: "#374151",
    marginBottom: 8,
    marginTop: 12,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse" as const,
    fontSize: 12,
  },
  th: {
    textAlign: "left" as const,
    padding: "8px 12px",
    backgroundColor: "#e5e7eb",
    color: "#374151",
    fontWeight: 600,
    borderBottom: "1px solid #d1d5db",
  },
  td: {
    padding: "8px 12px",
    borderBottom: "1px solid #e5e7eb",
    color: "#4b5563",
  },
  inputBar: {
    display: "flex",
    gap: 12,
    backgroundColor: "#ffffff",
    borderRadius: 12,
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
    padding: 16,
  },
  input: {
    flex: 1,
    padding: "12px 16px",
    border: "1px solid #d1d5db",
    borderRadius: 8,
    fontSize: 14,
    outline: "none",
  },
  button: {
    padding: "12px 24px",
    backgroundColor: "#6366f1",
    color: "#ffffff",
    border: "none",
    borderRadius: 8,
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: 8,
    transition: "background-color 0.15s ease",
  },
  buttonDisabled: {
    backgroundColor: "#a5b4fc",
    cursor: "not-allowed",
  },
  errorBanner: {
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#dc2626",
    padding: "12px 16px",
    borderRadius: 8,
    marginBottom: 12,
    fontSize: 13,
  },
  loadingDots: {
    display: "inline-flex",
    gap: 4,
  },
  loadingDot: {
    width: 6,
    height: 6,
    backgroundColor: "#6366f1",
    borderRadius: "50%",
    animation: "bounce 1.4s infinite ease-in-out both",
  },
};

// =============================================================================
// Helper Components
// =============================================================================

function SourceIcon({ type }: { type: string }) {
  if (type === "company") return <span style={styles.sourceIcon}>🏢</span>;
  if (type === "doc") return <span style={styles.sourceIcon}>📄</span>;
  return <span style={styles.sourceIcon}>📌</span>;
}

function StepPill({ step }: { step: Step }) {
  const statusStyle =
    step.status === "done"
      ? styles.stepDone
      : step.status === "running"
        ? styles.stepRunning
        : styles.stepPending;

  const statusIcon =
    step.status === "done" ? "✓" : step.status === "running" ? "⟳" : "○";

  return (
    <span style={{ ...styles.stepPill, ...statusStyle }}>
      {statusIcon} {step.label}
    </span>
  );
}

function DataTables({ rawData }: { rawData: RawData }) {
  const [expanded, setExpanded] = useState(false);

  const hasData =
    (rawData.companies && rawData.companies.length > 0) ||
    (rawData.activities && rawData.activities.length > 0) ||
    (rawData.opportunities && rawData.opportunities.length > 0);

  if (!hasData) return null;

  return (
    <div>
      <div
        style={styles.collapsibleHeader}
        onClick={() => setExpanded(!expanded)}
      >
        <span>{expanded ? "▼" : "▶"}</span>
        <span>Data used</span>
      </div>

      {expanded && (
        <div style={styles.dataSection}>
          {rawData.companies && rawData.companies.length > 0 && (
            <>
              <div style={{ ...styles.tableTitle, marginTop: 0 }}>
                Companies
              </div>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Name</th>
                    <th style={styles.th}>Plan</th>
                    <th style={styles.th}>Renewal Date</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.companies.map((c) => (
                    <tr key={c.company_id}>
                      <td style={styles.td}>{c.name}</td>
                      <td style={styles.td}>{c.plan}</td>
                      <td style={styles.td}>{c.renewal_date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {rawData.activities && rawData.activities.length > 0 && (
            <>
              <div style={styles.tableTitle}>Activities</div>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Type</th>
                    <th style={styles.th}>Occurred At</th>
                    <th style={styles.th}>Owner</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.activities.map((a) => (
                    <tr key={a.activity_id}>
                      <td style={styles.td}>{a.type}</td>
                      <td style={styles.td}>
                        {new Date(a.occurred_at).toLocaleString()}
                      </td>
                      <td style={styles.td}>{a.owner}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {rawData.opportunities && rawData.opportunities.length > 0 && (
            <>
              <div style={styles.tableTitle}>Opportunities</div>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Name</th>
                    <th style={styles.th}>Stage</th>
                    <th style={styles.th}>Expected Close</th>
                    <th style={styles.th}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.opportunities.map((o) => (
                    <tr key={o.opportunity_id}>
                      <td style={styles.td}>{o.name}</td>
                      <td style={styles.td}>{o.stage}</td>
                      <td style={styles.td}>{o.expected_close_date}</td>
                      <td style={styles.td}>
                        ${o.value.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function LoadingDots() {
  return (
    <span style={styles.loadingDots}>
      <span
        style={{ ...styles.loadingDot, animationDelay: "-0.32s" }}
      ></span>
      <span
        style={{ ...styles.loadingDot, animationDelay: "-0.16s" }}
      ></span>
      <span style={styles.loadingDot}></span>
    </span>
  );
}

function MessageBlock({ message }: { message: ChatMessage }) {
  return (
    <div style={styles.messageBlock}>
      <div style={styles.userLabel}>You</div>
      <div style={styles.userQuestion}>{message.question}</div>

      <div style={styles.assistantLabel}>Assistant</div>

      {message.response ? (
        <>
          <div style={styles.answerText}>{message.response.answer}</div>

          {message.response.sources && message.response.sources.length > 0 && (
            <div style={styles.sourcesRow}>
              <span style={{ fontSize: 12, color: "#6b7280", marginRight: 4 }}>
                Sources:
              </span>
              {message.response.sources.map((source) => (
                <span key={source.id} style={styles.sourceChip}>
                  <SourceIcon type={source.type} />
                  {source.label}
                </span>
              ))}
            </div>
          )}

          {message.response.meta && (
            <div style={styles.metaLine}>
              {message.response.meta.latency_ms && (
                <span>Latency: {message.response.meta.latency_ms}ms</span>
              )}
              {message.response.meta.latency_ms && message.response.meta.model && (
                <span> · </span>
              )}
              {message.response.meta.model && (
                <span>Model: {message.response.meta.model}</span>
              )}
            </div>
          )}

          {message.response.steps && message.response.steps.length > 0 && (
            <div style={styles.stepsRow}>
              <span style={{ fontSize: 12, color: "#6b7280", marginRight: 4 }}>
                Steps:
              </span>
              {message.response.steps.map((step) => (
                <StepPill key={step.id} step={step} />
              ))}
            </div>
          )}

          {message.response.raw_data && (
            <DataTables rawData={message.response.raw_data} />
          )}
        </>
      ) : (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <LoadingDots />
          <span style={{ fontSize: 14, color: "#6b7280" }}>Thinking…</span>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Main App Component
// =============================================================================

const EXAMPLE_PROMPTS = [
  "What's going on with Acme Manufacturing in the last 90 days?",
  "Which opportunities are close to renewing this month?",
  "Summarize recent activity for my largest accounts.",
];

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chatAreaRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const submitQuestion = async (question: string) => {
    if (!question.trim() || isLoading) return;

    setError(null);
    setIsLoading(true);

    const messageId = `msg-${Date.now()}`;
    const newMessage: ChatMessage = {
      id: messageId,
      question: question.trim(),
      response: null,
    };

    setMessages((prev) => [...prev, newMessage]);
    setCurrentQuestion("");

    try {
      const res = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: question.trim() }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error: ${res.status}`);
      }

      const data: ChatResponse = await res.json();

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId ? { ...msg, response: data } : msg
        )
      );
    } catch (err) {
      console.error("Chat API error:", err);
      setError(
        "Unable to reach the assistant. Please check that the backend is running."
      );
      // Remove the pending message on error
      setMessages((prev) => prev.filter((msg) => msg.id !== messageId));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitQuestion(currentQuestion);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !isLoading) {
      e.preventDefault();
      submitQuestion(currentQuestion);
    }
  };

  const handleSuggestionClick = (prompt: string) => {
    setCurrentQuestion(prompt);
    submitQuestion(prompt);
  };

  return (
    <div style={styles.page}>
      {/* CSS animation for loading dots */}
      <style>
        {`
          @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
          }
        `}
      </style>

      <div style={styles.container}>
        {/* Header */}
        <header style={styles.header}>
          <h1 style={styles.title}>Acme CRM AI Companion</h1>
          <p style={styles.subtitle}>
            Ask questions about your CRM accounts, activity, and pipeline.
          </p>
        </header>

        {/* Chat Area */}
        <div style={styles.chatArea} ref={chatAreaRef}>
          {messages.length === 0 ? (
            <div style={styles.emptyState}>
              <div style={styles.emptyTitle}>
                Try asking one of these questions:
              </div>
              <div style={styles.suggestionsContainer}>
                {EXAMPLE_PROMPTS.map((prompt, index) => (
                  <button
                    key={index}
                    style={styles.suggestionButton}
                    onClick={() => handleSuggestionClick(prompt)}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = "#e5e7eb";
                      e.currentTarget.style.borderColor = "#d1d5db";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "#f3f4f6";
                      e.currentTarget.style.borderColor = "#e5e7eb";
                    }}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div style={styles.messageList}>
              {messages.map((msg) => (
                <MessageBlock key={msg.id} message={msg} />
              ))}
            </div>
          )}
        </div>

        {/* Error Banner */}
        {error && <div style={styles.errorBanner}>{error}</div>}

        {/* Input Bar */}
        <form style={styles.inputBar} onSubmit={handleSubmit}>
          <input
            type="text"
            style={styles.input}
            placeholder="Ask a question about your CRM..."
            value={currentQuestion}
            onChange={(e) => setCurrentQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            type="submit"
            style={{
              ...styles.button,
              ...(isLoading ? styles.buttonDisabled : {}),
            }}
            disabled={isLoading || !currentQuestion.trim()}
          >
            {isLoading ? (
              <>
                <LoadingDots />
                <span>Thinking…</span>
              </>
            ) : (
              "Ask"
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
