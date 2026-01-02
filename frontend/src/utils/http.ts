/**
 * HTTP utilities for consistent error handling and response processing.
 * Consolidates duplicate patterns across hooks.
 */

/**
 * Check HTTP response and throw descriptive error if not ok
 */
export async function checkHttpResponse(response: Response): Promise<void> {
  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error");
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }
}

/**
 * Check if error is an abort error (should be ignored)
 */
export function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === "AbortError";
}

/**
 * Default error message for connection issues
 */
export const CONNECTION_ERROR_MESSAGE =
  "Unable to reach the assistant. Please check that the backend is running.";
