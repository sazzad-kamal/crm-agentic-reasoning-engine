/**
 * Types and configuration for DataExplorer component.
 */

// =============================================================================
// Types
// =============================================================================

export interface DataResponse {
  data: Record<string, unknown>[];
  total: number;
  columns: string[];
}

export type DataTab = "companies" | "contacts" | "opportunities" | "activities" | "groups" | "history";

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
    nestedFields: [
      { key: "_descriptions", label: "Description", icon: "📄" },
      { key: "_attachments", label: "Attachments", icon: "📎" }
    ]
  },
  { id: "activities", label: "Activities", icon: "📅", endpoint: "/api/data/activities" },
  { 
    id: "groups", 
    label: "Groups", 
    icon: "👥", 
    endpoint: "/api/data/groups",
    nestedFields: [{ key: "_members", label: "Members", icon: "🧑‍🤝‍🧑" }]
  },
  { id: "history", label: "History", icon: "📜", endpoint: "/api/data/history" },
];
