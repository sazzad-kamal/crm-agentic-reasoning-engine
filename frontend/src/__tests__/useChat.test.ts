import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useChat } from "../hooks/useChat";

// Mock fetch
const mockFetch = vi.fn();
(globalThis as typeof globalThis & { fetch: typeof fetch }).fetch = mockFetch;

describe("useChat", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns initial state", () => {
    const { result } = renderHook(() => useChat());

    expect(result.current.messages).toEqual([]);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("provides sendMessage, clearMessages, and clearError functions", () => {
    const { result } = renderHook(() => useChat());

    expect(typeof result.current.sendMessage).toBe("function");
    expect(typeof result.current.clearMessages).toBe("function");
    expect(typeof result.current.clearError).toBe("function");
  });

  it("does not send empty messages", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("");
    });

    expect(mockFetch).not.toHaveBeenCalled();
    expect(result.current.messages).toEqual([]);
  });

  it("does not send whitespace-only messages", async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("   ");
    });

    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("sets loading state while sending", async () => {
    mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves

    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.sendMessage("Test question");
    });

    expect(result.current.isLoading).toBe(true);
  });

  it("adds message optimistically", async () => {
    mockFetch.mockImplementation(() => new Promise(() => {}));

    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.sendMessage("Test question");
    });

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].question).toBe("Test question");
    expect(result.current.messages[0].response).toBeNull();
  });

  it("updates message with response on success", async () => {
    const mockResponse = {
      answer: "Test answer",
      sources: [{ type: "company", id: "1", label: "Acme" }],
      steps: [],
      raw_data: {},
      meta: { mode_used: "data", latency_ms: 100 },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("What is Acme?");
    });

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].response).toEqual(mockResponse);
    expect(result.current.isLoading).toBe(false);
  });

  it("sets error on API failure", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Internal Server Error",
    });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.error).toContain("500");
    expect(result.current.messages).toHaveLength(0); // Message removed on error
  });

  it("sets error on network failure", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network error"));

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.error).toBe("Network error");
    expect(result.current.messages).toHaveLength(0);
  });

  it("calls onError callback on failure", async () => {
    const onError = vi.fn();
    mockFetch.mockRejectedValueOnce(new Error("Test error"));

    const { result } = renderHook(() => useChat({ onError }));

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(onError).toHaveBeenCalledWith(expect.any(Error));
  });

  it("clears messages", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ answer: "Response", sources: [], steps: [] }),
    });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.messages).toHaveLength(1);

    act(() => {
      result.current.clearMessages();
    });

    expect(result.current.messages).toHaveLength(0);
  });

  it("clears error", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Error"));

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.error).not.toBeNull();

    act(() => {
      result.current.clearError();
    });

    expect(result.current.error).toBeNull();
  });

  it("does not allow sending while loading", async () => {
    mockFetch.mockImplementation(() => new Promise(() => {}));

    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.sendMessage("First");
    });

    expect(result.current.isLoading).toBe(true);

    await act(async () => {
      await result.current.sendMessage("Second");
    });

    // Should only have first message
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].question).toBe("First");
  });

  it("passes request options to API", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ answer: "Response", sources: [], steps: [] }),
    });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("Test", {
        mode: "docs",
        company_id: "acme-123",
      });
    });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        body: expect.stringContaining('"mode":"docs"'),
      })
    );
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        body: expect.stringContaining('"company_id":"acme-123"'),
      })
    );
  });

  it("trims question before sending", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ answer: "Response", sources: [], steps: [] }),
    });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("  Test question  ");
    });

    expect(result.current.messages[0].question).toBe("Test question");
  });

  it("generates unique message IDs", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ answer: "Response", sources: [], steps: [] }),
    });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.sendMessage("First");
    });
    await act(async () => {
      await result.current.sendMessage("Second");
    });

    const ids = result.current.messages.map((m) => m.id);
    expect(new Set(ids).size).toBe(2); // All unique
  });
});
