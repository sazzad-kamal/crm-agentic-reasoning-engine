// =============================================================================
// Configuration for Acme CRM AI Companion Frontend
// =============================================================================

// Get API URL from environment or default to same origin
const getApiUrl = (): string => {
  // Check for Vite env variable
  if (import.meta.env?.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  // In production (served from same origin), use empty string for relative URLs.
  // In development (Vite dev server on :5173), point to backend on :8000.
  return import.meta.env.DEV ? "http://localhost:8000" : "";
};

/**
 * API configuration
 * For local development: http://localhost:8000
 * For production: Set VITE_API_URL environment variable
 */
export const config = {
  // API base URL - defaults to localhost for local development
  apiUrl: getApiUrl(),
  
  // Feature flags
  features: {
    showDataTables: true,
    showFollowUpSuggestions: true,
  },
  
  // UI configuration
  ui: {
    maxMessagesInView: 50,
    animationDuration: 150,
  },
} as const;

/**
 * API endpoints
 */
export const endpoints = {
  chatStream: `${config.apiUrl}/api/chat/stream`,
  health: `${config.apiUrl}/api/health`,
  starterQuestions: `${config.apiUrl}/api/chat/starter-questions`,
} as const;

/**
 * Example prompts shown to new users (fallback if API unavailable).
 * Should match the starters in backend/agent/question_tree/
 */
export const EXAMPLE_PROMPTS = [
  "What deals are in the pipeline?",
  "Which accounts are up for renewal?",
  "What tasks are due this week?",
] as const;
