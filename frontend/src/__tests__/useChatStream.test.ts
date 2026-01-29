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
    // Message starts with empty answer and sectionStatus (shows skeleton loaders)
    expect(result.current.messages[0].response).toEqual({ answer: "" });
    expect(result.current.messages[0].sectionStatus).toEqual({
      data: "loading",
      answer: "loading",
      action: "loading",
      followup: "loading",
    });
  });

  it("parses done event and sets final response", async () => {
    const finalResponse = {
      answer: "Test answer",
      sources: [{ type: "company", id: "1", label: "Acme" }],
      steps: [],
      sql_results: {},
      meta: { mode_used: "data", latency_ms: 100 },
      follow_up_suggestions: ["What else?"],
    };

    const events = [
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
      `event: done\ndata: ${JSON.stringify({ answer: "Answer", sources: [], steps: [], sql_results: {}, meta: {} })}\n\n`,
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
      'event: answer_chunk\ndata: {"chunk": "Hello "}\n\n',
      'event: answer_chunk\ndata: {"chunk": "World"}\n\n',
      `event: done\ndata: ${JSON.stringify({ answer: "Hello World", sources: [], steps: [], sql_results: {}, meta: {} })}\n\n`,
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

  // ===========================================================================
  // Per-section SSE events
  // ===========================================================================

  it("handles data_ready event", async () => {
    const events = [
      'event: data_ready\ndata: {"sql_results": {"companies": [{"name": "Acme"}]}}\n\n',
      `event: done\ndata: ${JSON.stringify({ answer: "Done", sources: [], steps: [], sql_results: { companies: [{ name: "Acme" }] }, meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Show companies");
    });

    expect(result.current.messages[0].response?.sql_results).toEqual({
      companies: [{ name: "Acme" }],
    });
    expect(result.current.messages[0].sectionStatus).toBeUndefined();
  });

  it("handles action_ready event", async () => {
    const events = [
      'event: action_ready\ndata: {"suggested_action": "Schedule a call"}\n\n',
      `event: done\ndata: ${JSON.stringify({ answer: "Done", sources: [], steps: [], sql_results: {}, suggested_action: "Schedule a call", meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("What should I do?");
    });

    expect(result.current.messages[0].response?.suggested_action).toBe("Schedule a call");
  });

  it("handles action_ready with null action", async () => {
    const events = [
      'event: action_ready\ndata: {"suggested_action": null}\n\n',
      `event: done\ndata: ${JSON.stringify({ answer: "No action", sources: [], steps: [], sql_results: {}, suggested_action: null, meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Anything to do?");
    });

    expect(result.current.messages[0].response?.suggested_action).toBeNull();
  });

  it("handles followup_ready event", async () => {
    const events = [
      'event: followup_ready\ndata: {"follow_up_suggestions": ["Q1", "Q2"]}\n\n',
      `event: done\ndata: ${JSON.stringify({ answer: "Done", sources: [], steps: [], sql_results: {}, follow_up_suggestions: ["Q1", "Q2"], meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Tell me more");
    });

    expect(result.current.messages[0].response?.follow_up_suggestions).toEqual(["Q1", "Q2"]);
  });

  it("handles all section events together", async () => {
    const events = [
      'event: data_ready\ndata: {"sql_results": {"companies": []}}\n\n',
      'event: answer_chunk\ndata: {"chunk": "Here is "}\n\n',
      'event: answer_chunk\ndata: {"chunk": "the answer"}\n\n',
      'event: action_ready\ndata: {"suggested_action": "Follow up"}\n\n',
      'event: followup_ready\ndata: {"follow_up_suggestions": ["Next?"]}\n\n',
      `event: done\ndata: ${JSON.stringify({ answer: "Here is the answer", sources: [], steps: [], sql_results: { companies: [] }, suggested_action: "Follow up", follow_up_suggestions: ["Next?"], meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Full response");
    });

    const msg = result.current.messages[0];
    expect(msg.response?.answer).toBe("Here is the answer");
    expect(msg.response?.sql_results).toEqual({ companies: [] });
    expect(msg.response?.suggested_action).toBe("Follow up");
    expect(msg.response?.follow_up_suggestions).toEqual(["Next?"]);
    expect(msg.sectionStatus).toBeUndefined();
  });

  it("removes sectionStatus after done event", async () => {
    const events = [
      `event: done\ndata: ${JSON.stringify({ answer: "Answer", sources: [], steps: [], sql_results: {}, meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.messages[0].sectionStatus).toBeUndefined();
  });

  it("ignores unknown event types", async () => {
    const events = [
      'event: unknown_event\ndata: {"foo": "bar"}\n\n',
      `event: done\ndata: ${JSON.stringify({ answer: "Answer", sources: [], steps: [], sql_results: {}, meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.messages[0].response?.answer).toBe("Answer");
  });

  it("handles malformed SSE data gracefully", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const events = [
      "event: answer_chunk\ndata: {invalid json\n\n",
      `event: done\ndata: ${JSON.stringify({ answer: "Answer", sources: [], steps: [], sql_results: {}, meta: {} })}\n\n`,
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: createMockSSEStream(events),
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.messages[0].response?.answer).toBe("Answer");
    expect(warnSpy).toHaveBeenCalledWith("Failed to parse SSE data:", expect.any(String));
    warnSpy.mockRestore();
  });

  it("handles no response body", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: null,
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.error).toBe("No response body");
    expect(result.current.messages).toHaveLength(0);
  });

  it("handles network error with connection message", async () => {
    mockFetch.mockRejectedValueOnce("network failure");

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.error).toBe(
      "Unable to reach the assistant. Please check that the backend is running."
    );
  });

  it("handles HTTP error with text() failure", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      text: async () => { throw new Error("read failed"); },
    });

    const { result } = renderHook(() => useChatStream());

    await act(async () => {
      await result.current.sendMessage("Test");
    });

    expect(result.current.error).toBe("HTTP 503: Unknown error");
  });

  it("does not send when already loading", async () => {
    const neverEndingStream = new ReadableStream({ start() {} });

    mockFetch.mockResolvedValue({
      ok: true,
      body: neverEndingStream,
    });

    const { result } = renderHook(() => useChatStream());

    act(() => {
      result.current.sendMessage("First");
    });

    // Try sending while first is still loading
    await act(async () => {
      await result.current.sendMessage("Second");
    });

    // Only first message should be there
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].question).toBe("First");
  });

  it("abortStream stops loading and streaming", async () => {
    const neverEndingStream = new ReadableStream({ start() {} });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: neverEndingStream,
    });

    const { result } = renderHook(() => useChatStream());

    act(() => {
      result.current.sendMessage("Test");
    });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.isStreaming).toBe(true);

    act(() => {
      result.current.abortStream();
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.isStreaming).toBe(false);
  });
});
