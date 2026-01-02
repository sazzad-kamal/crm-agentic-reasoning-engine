import { describe, it, expect, vi } from "vitest";
import {
  checkHttpResponse,
  isAbortError,
  CONNECTION_ERROR_MESSAGE,
} from "../utils/http";

describe("http utilities", () => {
  // ===========================================================================
  // checkHttpResponse
  // ===========================================================================

  describe("checkHttpResponse", () => {
    it("does not throw for successful responses", async () => {
      const response = new Response("OK", { status: 200 });

      await expect(checkHttpResponse(response)).resolves.toBeUndefined();
    });

    it("does not throw for 201 Created", async () => {
      const response = new Response("Created", { status: 201 });

      await expect(checkHttpResponse(response)).resolves.toBeUndefined();
    });

    it("does not throw for 204 No Content", async () => {
      const response = new Response(null, { status: 204 });

      await expect(checkHttpResponse(response)).resolves.toBeUndefined();
    });

    it("throws for 400 Bad Request", async () => {
      const response = new Response("Invalid input", { status: 400 });

      await expect(checkHttpResponse(response)).rejects.toThrow(
        "HTTP 400: Invalid input"
      );
    });

    it("throws for 401 Unauthorized", async () => {
      const response = new Response("Authentication required", { status: 401 });

      await expect(checkHttpResponse(response)).rejects.toThrow(
        "HTTP 401: Authentication required"
      );
    });

    it("throws for 403 Forbidden", async () => {
      const response = new Response("Access denied", { status: 403 });

      await expect(checkHttpResponse(response)).rejects.toThrow(
        "HTTP 403: Access denied"
      );
    });

    it("throws for 404 Not Found", async () => {
      const response = new Response("Resource not found", { status: 404 });

      await expect(checkHttpResponse(response)).rejects.toThrow(
        "HTTP 404: Resource not found"
      );
    });

    it("throws for 500 Internal Server Error", async () => {
      const response = new Response("Server error", { status: 500 });

      await expect(checkHttpResponse(response)).rejects.toThrow(
        "HTTP 500: Server error"
      );
    });

    it("throws for 502 Bad Gateway", async () => {
      const response = new Response("Bad gateway", { status: 502 });

      await expect(checkHttpResponse(response)).rejects.toThrow(
        "HTTP 502: Bad gateway"
      );
    });

    it("throws for 503 Service Unavailable", async () => {
      const response = new Response("Service unavailable", { status: 503 });

      await expect(checkHttpResponse(response)).rejects.toThrow(
        "HTTP 503: Service unavailable"
      );
    });

    it("handles responses with empty body", async () => {
      const response = new Response("", { status: 500 });

      await expect(checkHttpResponse(response)).rejects.toThrow("HTTP 500: ");
    });

    it("handles responses where text() fails", async () => {
      const response = {
        ok: false,
        status: 500,
        text: vi.fn().mockRejectedValue(new Error("Body read failed")),
      } as unknown as Response;

      await expect(checkHttpResponse(response)).rejects.toThrow(
        "HTTP 500: Unknown error"
      );
    });

    it("handles JSON error responses", async () => {
      const errorJson = JSON.stringify({ error: "Validation failed" });
      const response = new Response(errorJson, { status: 422 });

      await expect(checkHttpResponse(response)).rejects.toThrow(
        `HTTP 422: ${errorJson}`
      );
    });
  });

  // ===========================================================================
  // isAbortError
  // ===========================================================================

  describe("isAbortError", () => {
    it("returns true for AbortError", () => {
      const abortError = new DOMException("Request aborted", "AbortError");

      expect(isAbortError(abortError)).toBe(true);
    });

    it("returns true for custom AbortError", () => {
      const customAbort = new Error("Aborted");
      customAbort.name = "AbortError";

      expect(isAbortError(customAbort)).toBe(true);
    });

    it("returns false for regular Error", () => {
      const regularError = new Error("Regular error");

      expect(isAbortError(regularError)).toBe(false);
    });

    it("returns false for TypeError", () => {
      const typeError = new TypeError("Type error");

      expect(isAbortError(typeError)).toBe(false);
    });

    it("returns false for string", () => {
      expect(isAbortError("AbortError")).toBe(false);
    });

    it("returns false for null", () => {
      expect(isAbortError(null)).toBe(false);
    });

    it("returns false for undefined", () => {
      expect(isAbortError(undefined)).toBe(false);
    });

    it("returns false for object with name property", () => {
      const fakeAbort = { name: "AbortError", message: "Fake abort" };

      expect(isAbortError(fakeAbort)).toBe(false);
    });

    it("returns false for error with similar name", () => {
      const similarError = new Error("Aborted");
      similarError.name = "AbortedError"; // Note: not "AbortError"

      expect(isAbortError(similarError)).toBe(false);
    });
  });

  // ===========================================================================
  // CONNECTION_ERROR_MESSAGE
  // ===========================================================================

  describe("CONNECTION_ERROR_MESSAGE", () => {
    it("is a non-empty string", () => {
      expect(typeof CONNECTION_ERROR_MESSAGE).toBe("string");
      expect(CONNECTION_ERROR_MESSAGE.length).toBeGreaterThan(0);
    });

    it("provides user-friendly message", () => {
      expect(CONNECTION_ERROR_MESSAGE).toContain("backend");
      expect(CONNECTION_ERROR_MESSAGE.toLowerCase()).toContain("running");
    });

    it("is exported and accessible", () => {
      expect(CONNECTION_ERROR_MESSAGE).toBe(
        "Unable to reach the assistant. Please check that the backend is running."
      );
    });
  });

  // ===========================================================================
  // Integration Scenarios
  // ===========================================================================

  describe("integration scenarios", () => {
    it("handles typical fetch error flow", async () => {
      // Simulate a 404 response
      const response = new Response("Not found", { status: 404 });

      try {
        await checkHttpResponse(response);
        expect.fail("Should have thrown");
      } catch (err) {
        expect(err).toBeInstanceOf(Error);
        expect((err as Error).message).toContain("404");
        expect(isAbortError(err)).toBe(false);
      }
    });

    it("handles abort controller cancellation", () => {
      const controller = new AbortController();
      controller.abort();

      // Simulate the error that would come from fetch
      const abortError = new DOMException("The operation was aborted", "AbortError");

      expect(isAbortError(abortError)).toBe(true);
    });

    it("handles network failures gracefully", () => {
      // Simulate a network error (TypeError from fetch)
      const networkError = new TypeError("Failed to fetch");

      expect(isAbortError(networkError)).toBe(false);
      expect(networkError.message).toBe("Failed to fetch");
    });
  });
});
