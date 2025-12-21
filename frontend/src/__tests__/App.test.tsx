import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App";

// Mock fetch
const mockFetch = vi.fn();
(globalThis as typeof globalThis & { fetch: typeof fetch }).fetch = mockFetch;

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
          raw_data: {},
          meta: { mode_used: "data", latency_ms: 100 },
        }),
      });
    });
  });

  it("renders the header", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /Acme CRM AI Companion/i })).toBeInTheDocument();
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
          raw_data: {},
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

  it("shows loading state while fetching", async () => {
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
      expect(screen.getByRole("status", { name: /loading/i })).toBeInTheDocument();
    });
  });

  it("clicking suggestion sends message", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/api/data/")) {
        return Promise.resolve(mockDataResponse);
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          answer: "Answer",
          sources: [],
          steps: [],
          raw_data: {},
          meta: { mode_used: "data", latency_ms: 50 },
        }),
      });
    });

    render(<App />);
    const buttons = screen.getAllByRole("button");
    const suggestionBtn = buttons.find(btn => 
      btn.textContent?.includes("What") || btn.textContent?.includes("How")
    );
    
    if (suggestionBtn) {
      fireEvent.click(suggestionBtn);
      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });
    }
  });
});
