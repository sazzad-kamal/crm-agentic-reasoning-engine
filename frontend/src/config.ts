// =============================================================================
// Configuration for Acme CRM AI Companion Frontend
// =============================================================================

// Get API URL from environment or default to localhost
const getApiUrl = (): string => {
  // Check for Vite env variable
  if (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) {
    return (import.meta as any).env.VITE_API_URL;
  }
  // Default for local development
  return "http://localhost:8000";
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
    showSteps: true,
    showFollowUpSuggestions: true,
    showSources: true,
    showLatency: true,
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
  chat: `${config.apiUrl}/api/chat`,
  health: `${config.apiUrl}/api/health`,
} as const;

/**
 * Example prompts shown to new users
 */
export const EXAMPLE_PROMPTS = [
  "What's going on with Acme Manufacturing in the last 90 days?",
  "Which opportunities are close to renewing this month?",
  "Summarize recent activity for my largest accounts.",
  "Show me the pipeline for TechCorp",
  "What renewals are coming up in the next 30 days?",
] as const;
