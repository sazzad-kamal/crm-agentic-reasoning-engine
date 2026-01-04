// =============================================================================
// Configuration for Acme CRM AI Companion Frontend
// =============================================================================

// Get API URL from environment or default to localhost
const getApiUrl = (): string => {
  // Check for Vite env variable
  if (import.meta.env?.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
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
  chatStream: `${config.apiUrl}/api/chat/stream`,
  health: `${config.apiUrl}/api/health`,
  starterQuestions: `${config.apiUrl}/api/data/starter-questions`,
} as const;

/**
 * Example prompts shown to new users (fallback if API unavailable).
 * Should match the starters in backend/agent/data/question_tree.json
 */
export const EXAMPLE_PROMPTS = [
  "How's my pipeline?",
  "Any renewals at risk?",
  "How's the team doing?",
] as const;
