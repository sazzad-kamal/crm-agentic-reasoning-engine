import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App";

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("App", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("renders the header", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /Acme CRM AI Companion/i })).toBeInTheDocument();
  });

  it("renders example prompts in empty state", () => {
    render(<App />);
    expect(screen.getByText(/Try asking one of these questions/i)).toBeInTheDocument();
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
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        answer: "Test answer",
        sources: [],
        steps: [],
        raw_data: {},
        meta: { mode_used: "data", latency_ms: 100 },
      }),
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask a question/i);
    
    fireEvent.change(input, { target: { value: "What is happening with Acme?" } });
    fireEvent.submit(input.closest("form")!);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  it("shows loading state while fetching", async () => {
    mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask a question/i);
    
    fireEvent.change(input, { target: { value: "Test" } });
    fireEvent.submit(input.closest("form")!);

    await waitFor(() => {
      expect(screen.getByRole("status", { name: /loading/i })).toBeInTheDocument();
    });
  });

  it("clicking suggestion sends message", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        answer: "Answer",
        sources: [],
        steps: [],
        raw_data: {},
        meta: { mode_used: "data", latency_ms: 50 },
      }),
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
