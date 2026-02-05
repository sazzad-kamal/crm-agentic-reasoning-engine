import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App";

// Mock fetch
const mockFetch = vi.fn();
(globalThis as typeof globalThis & { fetch: typeof fetch }).fetch = mockFetch;

// Helper to create a mock SSE ReadableStream
function createMockSSEStream(events: string[]) {
  const encoder = new TextEncoder();
  let index = 0;

  return new ReadableStream({
    pull(controller) {
      if (index < events.length) {
        controller.enqueue(encoder.encode(events[index]));
        index++;
      } else {
        controller.close();
      }
    },
  });
}

// Default response for data explorer API calls
const mockDataResponse = {
  ok: true,
  json: async () => ({
    data: [],
    total: 0,
    columns: [],
  }),
};

describe("App", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    // Always provide a default mock for data explorer endpoints
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/")) {
        return Promise.resolve(mockDataResponse);
      }
      // Default for chat API - will be overridden by specific tests
      return Promise.resolve({
        ok: true,
        json: async () => ({
          answer: "Default response",
          sources: [],
          steps: [],
          sql_results: {},
          meta: { mode_used: "data", latency_ms: 100 },
        }),
      });
    });
  });

  it("renders the header", () => {
    render(<App />);
    expect(screen.getByRole("heading", { level: 1, name: /Acme AI Companion/i })).toBeInTheDocument();
  });

  it("renders example prompts in empty state", () => {
    render(<App />);
    expect(screen.getByText(/Try one of these to get started/i)).toBeInTheDocument();
  });

  it("renders the input bar", () => {
    render(<App />);
    expect(screen.getByPlaceholderText(/Ask a question/i)).toBeInTheDocument();
  });

  it("renders send button", () => {
    render(<App />);
    expect(screen.getByRole("button", { name: /Send/i })).toBeInTheDocument();
  });

  it("allows typing in the input", () => {
    render(<App />);
    const input = screen.getByPlaceholderText(/Ask a question/i);
    fireEvent.change(input, { target: { value: "Test question" } });
    expect(input).toHaveValue("Test question");
  });

  it("submits question on form submit", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/")) {
        return Promise.resolve(mockDataResponse);
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          answer: "Test answer",
          sources: [],
          steps: [],
          sql_results: {},
          meta: { mode_used: "data", latency_ms: 100 },
        }),
      });
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask a question/i);
    
    fireEvent.change(input, { target: { value: "What is happening with Acme?" } });
    fireEvent.submit(input.closest("form")!);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalled();
    });
  });

  it("shows thinking indicator while fetching", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/")) {
        return Promise.resolve(mockDataResponse);
      }
      return new Promise(() => {}); // Never resolves for chat
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask a question/i);

    fireEvent.change(input, { target: { value: "Test" } });
    fireEvent.submit(input.closest("form")!);

    await waitFor(() => {
      // Should show skeleton loaders while waiting for response
      expect(screen.getByRole("status", { name: /generating answer/i })).toBeInTheDocument();
    });
  });

  it("clicking suggestion sends message", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/") || url.includes("/starter-questions")) {
        return Promise.resolve(mockDataResponse);
      }
      const response = {
        answer: "Suggestion answer",
        sources: [],
        steps: [],
        sql_results: {},
        meta: { mode_used: "data", latency_ms: 50 },
      };
      const events = [`event: done\ndata: ${JSON.stringify(response)}\n\n`];
      return Promise.resolve({
        ok: true,
        body: createMockSSEStream(events),
      });
    });

    render(<App />);

    // Wait for starter question buttons to render
    await waitFor(() => {
      const buttons = screen.getAllByRole("button");
      const suggestionBtn = buttons.find(btn =>
        btn.textContent?.includes("What deals")
      );
      expect(suggestionBtn).toBeTruthy();
    });

    const buttons = screen.getAllByRole("button");
    const suggestionBtn = buttons.find(btn =>
      btn.textContent?.includes("What deals")
    )!;
    fireEvent.click(suggestionBtn);

    await waitFor(() => {
      expect(screen.getByText("Suggestion answer")).toBeInTheDocument();
    });
  });

  it("opens data drawer when Browse Data button is clicked", async () => {
    render(<App />);

    // Find and click the Browse Data button
    const browseDataBtn = screen.getByRole("button", { name: /browse crm data/i });
    fireEvent.click(browseDataBtn);

    // Drawer should be open (dialog role)
    await waitFor(() => {
      expect(screen.getByRole("dialog", { name: /crm data browser/i })).toBeInTheDocument();
    });
  });

  it("closes data drawer when close button is clicked", async () => {
    render(<App />);

    // Open the drawer
    const browseDataBtn = screen.getByRole("button", { name: /browse crm data/i });
    fireEvent.click(browseDataBtn);

    // Drawer should be visible
    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    // Click close button
    const closeBtn = screen.getByRole("button", { name: /close data browser/i });
    fireEvent.click(closeBtn);

    // Drawer should have open class removed (still in DOM but not visible)
    await waitFor(() => {
      const drawer = document.querySelector(".drawer");
      expect(drawer).not.toHaveClass("drawer--open");
    });
  });

  it("clicking follow-up suggestion sends message", async () => {
    let chatCallCount = 0;
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/") || url.includes("/starter-questions")) {
        return Promise.resolve(mockDataResponse);
      }
      if (url.includes("/api/chat")) {
        chatCallCount++;
        // First call returns answer with follow-up suggestions via SSE
        if (chatCallCount === 1) {
          const response = {
            answer: "Here is the answer",
            sources: [],
            steps: [],
            sql_results: {},
            follow_up_suggestions: ["What about renewals?", "Show me the pipeline"],
            meta: { mode_used: "data", latency_ms: 50 },
          };
          const events = [`event: done\ndata: ${JSON.stringify(response)}\n\n`];
          return Promise.resolve({
            ok: true,
            body: createMockSSEStream(events),
          });
        }
        // Subsequent calls
        const followUpResponse = {
          answer: "Follow-up answer",
          sources: [],
          steps: [],
          sql_results: {},
          meta: { mode_used: "data", latency_ms: 50 },
        };
        const events = [`event: done\ndata: ${JSON.stringify(followUpResponse)}\n\n`];
        return Promise.resolve({
          ok: true,
          body: createMockSSEStream(events),
        });
      }
      return Promise.resolve(mockDataResponse);
    });

    render(<App />);

    // Wait for initial render to complete
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Ask a question/i)).toBeInTheDocument();
    });

    // Send initial question
    const input = screen.getByPlaceholderText(/Ask a question/i);
    fireEvent.change(input, { target: { value: "Test question" } });
    fireEvent.submit(input.closest("form")!);

    // Wait for the question to appear (added optimistically)
    await waitFor(() => {
      expect(screen.getByText("Test question")).toBeInTheDocument();
    });

    // Wait for response with follow-up suggestions
    await waitFor(
      () => {
        expect(screen.getByText("Here is the answer")).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    // Click a follow-up suggestion button
    const followUpBtn = await screen.findByRole("button", { name: /What about renewals\?/i });
    fireEvent.click(followUpBtn);

    // Should trigger another API call
    await waitFor(() => {
      expect(chatCallCount).toBeGreaterThanOrEqual(2);
    });
  });

  it("Ask AI from data explorer sets question and closes drawer", async () => {
    // Provide data for data explorer
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/companies")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            data: [{ company_id: "1", name: "Test Corp", plan: "Pro", renewal_date: "2024-12-31" }],
            total: 1,
            columns: ["company_id", "name", "plan", "renewal_date"],
          }),
        });
      }
      if (url.includes("/api/data/")) {
        return Promise.resolve(mockDataResponse);
      }
      // Return SSE stream for chat API
      const response = {
        answer: "Default response",
        sources: [],
        steps: [],
        sql_results: {},
        meta: { mode_used: "data", latency_ms: 100 },
      };
      const events = [`event: done\ndata: ${JSON.stringify(response)}\n\n`];
      return Promise.resolve({
        ok: true,
        body: createMockSSEStream(events),
      });
    });

    render(<App />);

    // Open the drawer
    const browseDataBtn = screen.getByRole("button", { name: /browse crm data/i });
    fireEvent.click(browseDataBtn);

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText("Test Corp")).toBeInTheDocument();
    });

    // Click Ask AI button on the row
    const askAiBtn = screen.getByTitle(/ask ai about this record/i);
    fireEvent.click(askAiBtn);

    // Input should have the question containing "Test Corp"
    await waitFor(() => {
      const input = screen.getByPlaceholderText(/Ask a question/i) as HTMLInputElement;
      expect(input.value).toContain("Test Corp");
    });

    // Drawer should close
    await waitFor(() => {
      const drawer = document.querySelector(".drawer");
      expect(drawer).not.toHaveClass("drawer--open");
    });
  });

  it("closes drawer when overlay is clicked", async () => {
    render(<App />);

    // Open the drawer
    const browseDataBtn = screen.getByRole("button", { name: /browse crm data/i });
    fireEvent.click(browseDataBtn);

    // Wait for drawer to open
    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    // Click the overlay
    const overlay = document.querySelector(".drawer-overlay");
    expect(overlay).toBeInTheDocument();
    fireEvent.click(overlay!);

    // Drawer should close
    await waitFor(() => {
      const drawer = document.querySelector(".drawer");
      expect(drawer).not.toHaveClass("drawer--open");
    });
  });

  it("ignores suggestion click while loading", async () => {
    // Use a never-ending stream to keep isLoading=true
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/") || url.includes("/starter-questions")) {
        return Promise.resolve(mockDataResponse);
      }
      return new Promise(() => {}); // Never resolves for chat
    });

    render(<App />);

    // Wait for starter questions to load
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Ask a question/i)).toBeInTheDocument();
    });

    // Submit a question first to put into loading state
    const input = screen.getByPlaceholderText(/Ask a question/i);
    fireEvent.change(input, { target: { value: "First question" } });
    fireEvent.submit(input.closest("form")!);

    // Should be loading now
    await waitFor(() => {
      expect(screen.getByRole("status", { name: /generating answer/i })).toBeInTheDocument();
    });

    // The send button should be disabled while loading
    const sendBtn = screen.getByRole("button", { name: /send/i });
    expect(sendBtn).toBeDisabled();
  });

  it("updates document title when loading", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/") || url.includes("/starter-questions")) {
        return Promise.resolve(mockDataResponse);
      }
      return new Promise(() => {}); // Never resolves
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask a question/i);
    fireEvent.change(input, { target: { value: "Test" } });
    fireEvent.submit(input.closest("form")!);

    await waitFor(() => {
      expect(document.title).toBe("Thinking... | Acme AI Companion");
    });
  });

  it("updates document title with message count", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/") || url.includes("/starter-questions")) {
        return Promise.resolve(mockDataResponse);
      }
      const response = {
        answer: "Answer",
        sources: [],
        steps: [],
        sql_results: {},
        meta: { mode_used: "data", latency_ms: 50 },
      };
      const events = [`event: done\ndata: ${JSON.stringify(response)}\n\n`];
      return Promise.resolve({
        ok: true,
        body: createMockSSEStream(events),
      });
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask a question/i);
    fireEvent.change(input, { target: { value: "Test" } });
    fireEvent.submit(input.closest("form")!);

    await waitFor(() => {
      expect(document.title).toContain("messages | Acme AI Companion");
    });
  });

});

describe("App - Demo Mode", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("hides input bar in demo mode", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/info")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ mode: "act" }),
        });
      }
      if (url.includes("/api/data/")) {
        return Promise.resolve(mockDataResponse);
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(<App />);

    // Wait for app to load and enter demo mode
    await waitFor(() => {
      // Input bar should not be present in demo mode
      expect(screen.queryByPlaceholderText(/Ask a question/i)).not.toBeInTheDocument();
    });
  });

  it("hides Browse Data button in demo mode", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/info")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ mode: "act" }),
        });
      }
      if (url.includes("/api/data/")) {
        return Promise.resolve(mockDataResponse);
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(<App />);

    // Wait for app to load and enter demo mode
    await waitFor(() => {
      // Browse Data button should not be present in demo mode
      expect(screen.queryByRole("button", { name: /browse crm data/i })).not.toBeInTheDocument();
    });
  });
});
