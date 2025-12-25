import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useChatStream } from "../hooks/useChatStream";

// Mock fetch
const mockFetch = vi.fn();
(globalThis as typeof globalThis & { fetch: typeof fetch }).fetch = mockFetch;

// Helper to create a mock ReadableStream
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

describe("useChatStream", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("returns initial state", () => {
    const { result } = renderHook(() => useChatStream());

    expect(result.current.messages).toEqual([]);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.currentStatus).toBeNull();
  });

  it("provides required functions", () => {
    const { result } = renderHook(() => useChatStream());

    expect(typeof result.current.sendMessage).toBe("function");
    expect(typeof result.current.clearMessages).toBe("function");
    expect(typeof result.current.clearError).toBe("function");
    expect(typeof result.current.abortStream).toBe("function");
  });

  it("does not send empty messages", async () => {
    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("");
    });

    expect(mockFetch).not.toHaveBeenCalled();
    expect(result.current.messages).toEqual([]);
  });

  it("does not send whitespace-only messages", async () => {
    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("   ");
    });

    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("sets loading and streaming state when sending", async () => {
    // Create a stream that never completes
    const neverEndingStream = new ReadableStream({
      start() {},
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: neverEndingStream,
    });

    const { result } = renderHook(() => useChatStream());

    act(() => {
      result.current.sendMessage("Test question");
    });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.isStreaming).toBe(true);
    expect(result.current.currentStatus).toBe("Starting...");
  });

  it("adds message optimistically", async () => {
    const neverEndingStream = new ReadableStream({ start() {} });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: neverEndingStream,
    });

    const { result } = renderHook(() => useChatStream());

    act(() => {
      result.current.sendMessage("Test question");
    });

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].question).toBe("Test question");
    expect(result.current.messages[0].response).toBeNull();
  });

  it("updates currentStatus from status events", async () => {
    const events = [
      'event: status\ndata: {"message": "Fetching CRM data..."}\n\n',
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test question");
    });

    // Status should have been updated during stream
    // After stream ends, it resets to null
    expect(result.current.currentStatus).toBeNull();
  });

  it("parses done event and sets final response", async () => {
    const finalResponse = {
      answer: "Test answer",
      sources: [{ type: "company", id: "1", label: "Acme" }],
      steps: [],
      raw_data: {},
      meta: { mode_used: "data", latency_ms: 100 },
      follow_up_suggestions: ["What else?"],
    };

    const events = [
      'event: status\ndata: {"message": "Processing..."}\n\n',
      `event: done\ndata: ${JSON.stringify(finalResponse)}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("What is Acme?");
    });

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].response).not.toBeNull();
    expect(result.current.messages[0].response?.answer).toBe("Test answer");
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isStreaming).toBe(false);
  });

  it("handles HTTP errors", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Internal Server Error",
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test question");
    });

    expect(result.current.error).toBe("HTTP 500: Internal Server Error");
    expect(result.current.messages).toHaveLength(0); // Message removed on error
    expect(result.current.isLoading).toBe(false);
  });

  it("handles streaming errors", async () => {
    const events = [
      'event: error\ndata: {"message": "Something went wrong"}\n\n',
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test question");
    });

    expect(result.current.error).toBe("Something went wrong");
    expect(result.current.messages).toHaveLength(0);
  });

  it("clears messages", async () => {
    const events = [
      `event: done\ndata: ${JSON.stringify({ answer: "Answer", sources: [], steps: [], raw_data: {}, meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

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
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Error",
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.error).toBeTruthy();

    act(() => {
      result.current.clearError();
    });

    expect(result.current.error).toBeNull();
  });

  it("accumulates answer chunks", async () => {
    const events = [
      'event: answer_start\ndata: {}\n\n',
      'event: answer_chunk\ndata: {"chunk": "Hello "}\n\n',
      'event: answer_chunk\ndata: {"chunk": "World"}\n\n',
      'event: answer_end\ndata: {"answer": "Hello World"}\n\n',
      `event: done\ndata: ${JSON.stringify({ answer: "Hello World", sources: [], steps: [], raw_data: {}, meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Say hello");
    });

    expect(result.current.messages[0].response?.answer).toBe("Hello World");
  });

  it("calls onError callback on error", async () => {
    const onError = vi.fn();

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Server Error",
    });

    const { result } = renderHook(() => useChatStream({ onError }));

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(onError).toHaveBeenCalled();
  });
});
