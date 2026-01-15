/**
 * Types and configuration for DataExplorer component.
 * Centralized in types/ for proper dependency flow.
 */

// =============================================================================
// Types
// =============================================================================

export interface DataResponse {
  data: Record<string, unknown>[];
  total: number;
  columns: string[];
}

export type DataTab = "companies" | "contacts" | "opportunities" | "activities" | "history";

export interface NestedFieldConfig {
  key: string;
  label: string;
  icon: string;
}

export interface TabConfig {
  id: DataTab;
  label: string;
  icon: string;
  endpoint: string;
  nestedFields?: NestedFieldConfig[];
}

// =============================================================================
// Configuration
// =============================================================================

export const TABS: TabConfig[] = [
  {
    id: "companies",
    label: "Companies",
    icon: "🏢",
    endpoint: "/api/data/companies",
    nestedFields: [{ key: "_private_texts", label: "Notes & Attachments", icon: "📝" }]
  },
  {
    id: "contacts",
    label: "Contacts",
    icon: "👤",
    endpoint: "/api/data/contacts",
    nestedFields: [{ key: "_private_texts", label: "Notes & Attachments", icon: "📝" }]
  },
  {
    id: "opportunities",
    label: "Opportunities",
    icon: "💰",
    endpoint: "/api/data/opportunities",
    nestedFields: [{ key: "_private_texts", label: "Notes", icon: "📝" }]
  },
  { id: "activities", label: "Activities", icon: "📅", endpoint: "/api/data/activities" },
  { id: "history", label: "History", icon: "📜", endpoint: "/api/data/history" },
];
