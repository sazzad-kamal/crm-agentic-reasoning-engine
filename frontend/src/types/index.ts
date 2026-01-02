// =============================================================================
// TypeScript Types for Acme CRM AI Companion
// =============================================================================

export interface Source {
  type: "company" | "doc" | "activity" | "opportunity" | "history";
  id: string;
  label: string;
}

export interface Step {
  id: string;
  label: string;
  status: "done" | "pending" | "running" | "error" | "skipped";
}

// Internal types for RawData (not exported - only used within this module)
interface Company {
  company_id: string;
  name: string;
  plan: string;
  renewal_date: string;
}

interface Activity {
  activity_id: string;
  type: string;
  occurred_at: string;
  owner: string;
  summary?: string;
  subject?: string;
}

interface Opportunity {
  opportunity_id: string;
  name: string;
  stage: string;
  expected_close_date: string;
  value: number;
}

interface HistoryEntry {
  history_id: string;
  event_type: string;
  occurred_at: string;
  description?: string;
}

export interface RawData {
  companies?: Company[];
  activities?: Activity[];
  opportunities?: Opportunity[];
  history?: HistoryEntry[];
  renewals?: Array<{
    company_id: string;
    company_name: string;
    renewal_date: string;
    plan: string;
  }>;
  pipeline_summary?: {
    total_value: number;
    count: number;
    stages: Record<string, number>;
  };
}

export interface Meta {
  mode_used?: string;
  latency_ms?: number;
  model?: string;
  company_id?: string;
  days?: number;
}

export interface ChatResponse {
  answer: string;
  sources?: Source[];
  steps?: Step[];
  raw_data?: RawData;
  meta?: Meta;
  follow_up_suggestions?: string[];
}

export interface ChatMessage {
  id: string;
  question: string;
  response: ChatResponse | null;
  timestamp: Date;
}

export interface ChatRequest {
  question: string;
  mode?: "auto" | "docs" | "data" | "data+docs";
  company_id?: string;
  session_id?: string;
  user_id?: string;
}
