// =============================================================================
// TypeScript Types for Acme CRM AI Companion
// =============================================================================

// Nested data types
interface PrivateText {
  id: string;
  type: string;
  title: string;
  text: string;
  company_id?: string;
  contact_id?: string;
  opportunity_id?: string;
  metadata?: {
    created_at?: string;
    file_type?: string;
    history_type?: string;
    [key: string]: unknown;
  };
}

// Internal types for RawData (not exported - only used within this module)
interface Company {
  company_id: string;
  name: string;
  plan: string;
  renewal_date: string;
  _private_texts?: PrivateText[];
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
  notes?: string;
  _private_texts?: PrivateText[];
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

export interface ChatResponse {
  answer: string;
  sql_results?: RawData;
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
  session_id?: string;
}
