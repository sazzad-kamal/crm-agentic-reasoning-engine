/**
 * Tests for useDataFetch hook.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useDataFetch } from "../hooks/useDataFetch";

// Mock fetch
const mockFetch = vi.fn();
(globalThis as typeof globalThis & { fetch: typeof fetch }).fetch = mockFetch;

// Mock config
vi.mock("../config", () => ({
  config: {
    apiUrl: "http://localhost:8000",
  },
}));

describe("useDataFetch", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("initial state", () => {
    it("starts with loading true", () => {
      mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves
      
      const { result } = renderHook(() => useDataFetch("/api/data/companies"));
      
      expect(result.current.loading).toBe(true);
      expect(result.current.data).toBeNull();
      expect(result.current.error).toBeNull();
    });
  });

  describe("successful fetch", () => {
    it("fetches data and sets loading to false", async () => {
      const mockData = {
        data: [{ id: 1, name: "Test Company" }],
        total: 1,
        columns: ["id", "name"],
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockData),
      });

      const { result } = renderHook(() => useDataFetch("/api/data/companies"));

      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });

      expect(result.current.data).toEqual(mockData);
      expect(result.current.error).toBeNull();
    });

    it("calls fetch with correct URL", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ data: [], total: 0, columns: [] }),
      });

      renderHook(() => useDataFetch("/api/data/contacts"));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          "http://localhost:8000/api/data/contacts",
          expect.objectContaining({ signal: expect.any(AbortSignal) })
        );
      });
    });
  });

  describe("error handling", () => {
    it("handles HTTP error responses", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      const { result } = renderHook(() => useDataFetch("/api/data/companies"));

      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });

      expect(result.current.error).toBe("HTTP 500");
      expect(result.current.data).toBeNull();
    });

    it("handles network errors", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network failure"));

      const { result } = renderHook(() => useDataFetch("/api/data/companies"));

      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });

      expect(result.current.error).toBe("Network failure");
      expect(result.current.data).toBeNull();
    });

    it("handles 404 errors", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const { result } = renderHook(() => useDataFetch("/api/data/unknown"));

      await waitFor(() => {
        expect(result.current.error).toBe("HTTP 404");
      });
    });
  });

  describe("endpoint changes", () => {
    it("refetches when endpoint changes", async () => {
      const companiesData = { data: [{ id: 1 }], total: 1, columns: ["id"] };
      const contactsData = { data: [{ id: 2 }], total: 1, columns: ["id"] };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(companiesData),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(contactsData),
        });

      const { result, rerender } = renderHook(
        ({ endpoint }) => useDataFetch(endpoint),
        { initialProps: { endpoint: "/api/data/companies" } }
      );

      await waitFor(() => {
        expect(result.current.data).toEqual(companiesData);
      });

      rerender({ endpoint: "/api/data/contacts" });

      await waitFor(() => {
        expect(result.current.data).toEqual(contactsData);
      });

      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it("sets loading true when endpoint changes", async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: [], total: 0, columns: [] }),
        })
        .mockImplementation(() => new Promise(() => {})); // Second call never resolves

      const { result, rerender } = renderHook(
        ({ endpoint }) => useDataFetch(endpoint),
        { initialProps: { endpoint: "/api/data/companies" } }
      );

      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });

      rerender({ endpoint: "/api/data/contacts" });

      // With batched state updates, loading is set to true on initial render
      // After rerender with new endpoint, the effect triggers and loading becomes true
      await waitFor(() => {
        expect(result.current.loading).toBe(true);
      });
    });
  });

  describe("cleanup", () => {
    it("cancels pending requests on unmount", async () => {
      let resolvePromise: (value: unknown) => void;
      const pendingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      mockFetch.mockReturnValueOnce(pendingPromise);

      const { result, unmount } = renderHook(() => useDataFetch("/api/data/companies"));

      expect(result.current.loading).toBe(true);

      unmount();

      // Resolve after unmount - should not update state
      resolvePromise!({
        ok: true,
        json: () => Promise.resolve({ data: [], total: 0, columns: [] }),
      });

      // No error should occur (React would warn about state updates on unmounted components)
      // The cancelled flag prevents this
    });

    it("clears error when refetching", async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ data: [], total: 0, columns: [] }),
        });

      const { result, rerender } = renderHook(
        ({ endpoint }) => useDataFetch(endpoint),
        { initialProps: { endpoint: "/api/data/companies" } }
      );

      await waitFor(() => {
        expect(result.current.error).toBe("HTTP 500");
      });

      rerender({ endpoint: "/api/data/contacts" });

      // Error should be cleared while loading new data
      await waitFor(() => {
        expect(result.current.error).toBeNull();
      });
    });
  });
});
