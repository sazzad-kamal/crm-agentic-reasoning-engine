import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DataTables } from "../components/DataTables";
import type { RawData } from "../types";

// Mock the keyboard utility
vi.mock("../utils/keyboard", () => ({
  createActivationHandler: (fn: () => void) => (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fn();
    }
  },
}));

describe("DataTables", () => {
  // =========================================================================
  // Empty State
  // =========================================================================

  it("returns null when no data", () => {
    const rawData: RawData = {};
    const { container } = render(<DataTables rawData={rawData} />);

    expect(container.firstChild).toBeNull();
  });

  it("returns null when all arrays are empty", () => {
    const rawData: RawData = {
      companies: [],
      activities: [],
      opportunities: [],
      history: [],
      renewals: [],
    };
    const { container } = render(<DataTables rawData={rawData} />);

    expect(container.firstChild).toBeNull();
  });

  // =========================================================================
  // Default Collapsed State
  // =========================================================================

  it("renders collapsed by default", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

    expect(screen.getByText(/Data used/)).toBeInTheDocument();
    expect(screen.queryByText("Acme")).not.toBeInTheDocument();
  });

  it("shows data types as icons for single table", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Basic", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

    expect(screen.getByText("Data used")).toBeInTheDocument();
    expect(screen.getByTitle("companies (1)")).toBeInTheDocument();
  });

  it("shows data types as icons for multiple tables", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
      activities: [
        {
          activity_id: "1",
          type: "Meeting",
          occurred_at: "2024-01-15T10:00:00",
          owner: "John",
        },
      ],
    };

    render(<DataTables rawData={rawData} />);

    expect(screen.getByText("Data used")).toBeInTheDocument();
    expect(screen.getByTitle("companies (1)")).toBeInTheDocument();
    expect(screen.getByTitle("activities (1)")).toBeInTheDocument();
  });

  it("shows expand arrow when collapsed by default", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

    expect(screen.getByText("▶")).toBeInTheDocument();
  });

  // =========================================================================
  // Toggle State
  // =========================================================================

  it("expands when header clicked", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme Corp", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    expect(screen.queryByText("Acme Corp")).not.toBeInTheDocument();

    const header = screen.getByRole("button");
    fireEvent.click(header);

    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  it("shows collapse arrow when expanded", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

    const header = screen.getByRole("button");
    fireEvent.click(header);

    expect(screen.getByText("▼")).toBeInTheDocument();
  });

  it("re-collapses when header clicked again", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    expect(screen.queryByText("Acme")).not.toBeInTheDocument();

    const header = screen.getByRole("button");
    fireEvent.click(header);
    expect(screen.getByText("Acme")).toBeInTheDocument();

    fireEvent.click(header);
    expect(screen.queryByText("Acme")).not.toBeInTheDocument();
  });

  // =========================================================================
  // Companies Table
  // =========================================================================

  it("renders companies table", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme Corp", plan: "Enterprise", renewal_date: "2024-12-31" },
        { company_id: "2", name: "TechCo", plan: "Pro", renewal_date: "2024-11-30" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Companies (2)")).toBeInTheDocument();
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
    expect(screen.getByText("TechCo")).toBeInTheDocument();
    expect(screen.getByText("Enterprise")).toBeInTheDocument();
    expect(screen.getByText("Pro")).toBeInTheDocument();
  });

  it("companies table has correct headers", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Basic", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Plan")).toBeInTheDocument();
    expect(screen.getByText("Renewal Date")).toBeInTheDocument();
  });

  // =========================================================================
  // Activities Table
  // =========================================================================

  it("renders activities table", () => {
    const rawData: RawData = {
      activities: [
        {
          activity_id: "1",
          type: "Call",
          occurred_at: "2024-01-15T10:00:00",
          owner: "John Doe",
          summary: "Discussed contract",
        },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Activities (1)")).toBeInTheDocument();
    expect(screen.getByText("Call")).toBeInTheDocument();
    expect(screen.getByText("John Doe")).toBeInTheDocument();
    expect(screen.getByText("Discussed contract")).toBeInTheDocument();
  });

  it("activities table uses subject when summary not available", () => {
    const rawData: RawData = {
      activities: [
        {
          activity_id: "1",
          type: "Email",
          occurred_at: "2024-01-15T10:00:00",
          owner: "Jane",
          subject: "Follow-up email",
        },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Follow-up email")).toBeInTheDocument();
  });

  it("activities table shows dash when no summary or subject", () => {
    const rawData: RawData = {
      activities: [
        {
          activity_id: "1",
          type: "Meeting",
          occurred_at: "2024-01-15T10:00:00",
          owner: "Bob",
        },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    const cells = screen.getAllByText("—");
    expect(cells.length).toBeGreaterThan(0);
  });

  // =========================================================================
  // Opportunities Table
  // =========================================================================

  it("renders opportunities table", () => {
    const rawData: RawData = {
      opportunities: [
        {
          opportunity_id: "1",
          name: "Q1 Deal",
          stage: "Negotiation",
          expected_close_date: "2024-03-31",
          value: 50000,
        },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Opportunities (1)")).toBeInTheDocument();
    expect(screen.getByText("Q1 Deal")).toBeInTheDocument();
    expect(screen.getByText("Negotiation")).toBeInTheDocument();
    expect(screen.getByText(/50,000/)).toBeInTheDocument(); // Currency formatted
  });

  // =========================================================================
  // History Table
  // =========================================================================

  it("renders history table", () => {
    const rawData: RawData = {
      history: [
        {
          history_id: "1",
          event_type: "Status Change",
          occurred_at: "2024-01-15T10:00:00",
          notes: "Changed from Lead to Customer",
        },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("History (1)")).toBeInTheDocument();
    expect(screen.getByText("Status Change")).toBeInTheDocument();
    expect(screen.getByText("Changed from Lead to Customer")).toBeInTheDocument();
  });

  // =========================================================================
  // Renewals Table
  // =========================================================================

  it("renders renewals table", () => {
    const rawData: RawData = {
      renewals: [
        {
          company_id: "1",
          company_name: "Acme Corp",
          renewal_date: "2024-06-30",
          plan: "Enterprise",
        },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Upcoming Renewals (1)")).toBeInTheDocument();
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
    expect(screen.getByText("Enterprise")).toBeInTheDocument();
  });

  // =========================================================================
  // Pipeline Summary
  // =========================================================================

  it("renders pipeline summary", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
      pipeline_summary: {
        total_value: 250000,
        count: 15,
        stages: {
          Prospecting: 5,
          Negotiation: 10,
        },
      },
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Pipeline Summary")).toBeInTheDocument();
    expect(screen.getByText("Total Value")).toBeInTheDocument();
    expect(screen.getByText(/250,000/)).toBeInTheDocument();
    expect(screen.getByText("Opportunity Count")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
  });

  // =========================================================================
  // Multiple Tables
  // =========================================================================

  it("renders all table types together", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme", plan: "Pro", renewal_date: "2024-12-31" },
      ],
      activities: [
        {
          activity_id: "1",
          type: "Call",
          occurred_at: "2024-01-15T10:00:00",
          owner: "John",
        },
      ],
      opportunities: [
        {
          opportunity_id: "1",
          name: "Deal",
          stage: "Closed Won",
          expected_close_date: "2024-03-31",
          value: 10000,
        },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Companies (1)")).toBeInTheDocument();
    expect(screen.getByText("Activities (1)")).toBeInTheDocument();
    expect(screen.getByText("Opportunities (1)")).toBeInTheDocument();
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("has proper ARIA attributes", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

    const section = screen.getByLabelText("Data used in response");
    expect(section).toBeInTheDocument();

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(button).toHaveAttribute("aria-controls", "data-tables-content");
  });

  it("updates aria-expanded when toggled", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("tables have proper aria-labelledby", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    const table = screen.getByLabelText("Companies data").querySelector("table");
    expect(table).toHaveAttribute("aria-labelledby", "companies-table-label");
  });

  // =========================================================================
  // Keyboard Navigation
  // =========================================================================

  it("expands on Enter key", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    expect(screen.queryByText("Acme")).not.toBeInTheDocument();

    const button = screen.getByRole("button");
    fireEvent.keyDown(button, { key: "Enter" });

    expect(screen.getByText("Acme")).toBeInTheDocument();
  });

  it("expands on Space key", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    expect(screen.queryByText("Acme")).not.toBeInTheDocument();

    const button = screen.getByRole("button");
    fireEvent.keyDown(button, { key: " " });

    expect(screen.getByText("Acme")).toBeInTheDocument();
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("memoizes correctly", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    const { rerender } = render(<DataTables rawData={rawData} />);

    rerender(<DataTables rawData={rawData} />);
    expect(screen.getByText("Data used")).toBeInTheDocument();
  });

  // =========================================================================
  // Generic Data Table (rawData.data)
  // =========================================================================

  it("renders generic data table when rawData.data is provided", () => {
    const rawData: RawData = {
      data: [
        { name: "Alice", role: "Engineer", department: "R&D" },
        { name: "Bob", role: "Designer", department: "UX" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Results (2)")).toBeInTheDocument();
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("Engineer")).toBeInTheDocument();
    expect(screen.getByText("Designer")).toBeInTheDocument();
  });

  it("generic data table filters out columns starting with underscore", () => {
    const rawData: RawData = {
      data: [
        { name: "Alice", _internal_id: "x1", role: "Engineer" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("name")).toBeInTheDocument();
    expect(screen.getByText("role")).toBeInTheDocument();
    expect(screen.queryByText("_internal_id")).not.toBeInTheDocument();
    expect(screen.queryByText("x1")).not.toBeInTheDocument();
  });

  it("generic data table replaces underscores in column headers", () => {
    const rawData: RawData = {
      data: [
        { first_name: "Alice", last_name: "Smith" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("first name")).toBeInTheDocument();
    expect(screen.getByText("last name")).toBeInTheDocument();
  });

  it("generic data table renders null values as em dash", () => {
    const rawData: RawData = {
      data: [
        { name: "Alice", email: null },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Alice")).toBeInTheDocument();
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThan(0);
  });

  // =========================================================================
  // Grouped Source Tables (UNION ALL with source column)
  // =========================================================================

  it("renders grouped tables when data has source column", () => {
    const rawData: RawData = {
      data: [
        { source: "company", name: "Crown Foods", plan: "Enterprise", status: "Active", health_status: "At Risk", key_date: "2025-03-15", notes: "Key account" },
        { source: "activity", name: "Call", plan: "Renewal check-in", status: "Open", health_status: "High", key_date: "2025-02-05", notes: "Prep agenda" },
        { source: "contact", name: "Maria Lopez", plan: "Decision Maker", status: "VP Operations", health_status: "maria@crown.com", key_date: "", notes: "Primary contact" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    // Grouped table titles
    expect(screen.getByText(/Company \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/Activities \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/Contacts \(1\)/)).toBeInTheDocument();

    // Company data
    expect(screen.getByText("Crown Foods")).toBeInTheDocument();
    expect(screen.getByText("Enterprise")).toBeInTheDocument();

    // Activity data
    expect(screen.getByText("Renewal check-in")).toBeInTheDocument();

    // Contact data
    expect(screen.getByText("Maria Lopez")).toBeInTheDocument();
    expect(screen.getByText("Decision Maker")).toBeInTheDocument();
  });

  it("renders grouped opportunity and history tables", () => {
    const rawData: RawData = {
      data: [
        { source: "opportunity", name: "Q1 Renewal", plan: "Negotiation", status: "Renewal", health_status: "$50,000", key_date: "2025-03-31", notes: "" },
        { source: "history", name: "Status Change", plan: "Moved to At Risk", key_date: "2025-01-20", notes: "Health downgraded" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText(/Opportunities \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/History \(1\)/)).toBeInTheDocument();
    expect(screen.getByText("Q1 Renewal")).toBeInTheDocument();
    expect(screen.getByText("Status Change")).toBeInTheDocument();
  });

  it("skips unknown source types in grouped tables", () => {
    const rawData: RawData = {
      data: [
        { source: "company", name: "Acme", plan: "Pro", status: "Active", health_status: "Good", key_date: "2025-06-01", notes: "" },
        { source: "unknown_type", name: "Mystery", plan: "N/A", status: "N/A", health_status: "N/A", key_date: "", notes: "" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText(/Company \(1\)/)).toBeInTheDocument();
    expect(screen.queryByText("Mystery")).not.toBeInTheDocument();
  });

  it("truncates long notes in grouped tables", () => {
    const longNote = "A".repeat(100);
    const rawData: RawData = {
      data: [
        { source: "company", name: "Test Co", plan: "Basic", status: "Active", health_status: "Good", key_date: "2025-12-31", notes: longNote },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    // Should truncate at 80 chars with ellipsis
    const truncated = "A".repeat(80) + "…";
    expect(screen.getByText(truncated)).toBeInTheDocument();
  });

  it("shows em dash for missing values in grouped tables", () => {
    const rawData: RawData = {
      data: [
        { source: "contact", name: "John Doe", plan: "Decision Maker", status: "CTO", health_status: null, key_date: "", notes: "" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("John Doe")).toBeInTheDocument();
    // null health_status renders as "—"
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThan(0);
  });

  it("grouped tables have proper ARIA attributes", () => {
    const rawData: RawData = {
      data: [
        { source: "company", name: "Acme", plan: "Pro", status: "Active", health_status: "Good", key_date: "2025-06-01", notes: "" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    const region = screen.getByLabelText("Company data");
    expect(region).toBeInTheDocument();
    expect(region).toHaveAttribute("data-type", "company");

    const table = region.querySelector("table");
    expect(table).toHaveAttribute("aria-labelledby", "grouped-company-table-label");
  });

  // =========================================================================
  // Dynamic Entity Groups (Demo Mode)
  // =========================================================================

  it("renders dynamic array entities from demo mode", () => {
    const rawData: RawData = {
      today_meetings: [
        { name: "Call with John", time: "10:00 AM" },
        { name: "Team standup", time: "9:00 AM" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Today's Meetings (2)")).toBeInTheDocument();
    expect(screen.getByText("Call with John")).toBeInTheDocument();
    expect(screen.getByText("Team standup")).toBeInTheDocument();
  });

  it("renders scalar metrics from demo mode", () => {
    const rawData: RawData = {
      total_pipeline: 500000,
      at_risk_pct: 15.5,
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Summary Metrics")).toBeInTheDocument();
    expect(screen.getByText("Total Pipeline")).toBeInTheDocument();
    expect(screen.getByText("At-Risk %")).toBeInTheDocument();
  });

  it("renders nested forecast object with deals array", () => {
    const rawData: RawData = {
      forecast_30d: {
        deals: [
          { name: "Deal A", value: 10000 },
          { name: "Deal B", value: 20000 },
        ],
        weighted: 25000,
        count: 2,
      },
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    // Should show the deals array as a table
    expect(screen.getByText("30-Day Forecast (2)")).toBeInTheDocument();
    expect(screen.getByText("Deal A")).toBeInTheDocument();
    expect(screen.getByText("Deal B")).toBeInTheDocument();

    // Should show numeric summaries as metrics
    expect(screen.getByText("Summary Metrics")).toBeInTheDocument();
  });

  it("renders at-risk deals with proper formatting", () => {
    const rawData: RawData = {
      at_risk_deals: [
        { name: "Stalled Deal", risk_reason: "Stalled", value: 50000, probability: 60 },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("At-Risk Deals (1)")).toBeInTheDocument();
    expect(screen.getByText("Stalled Deal")).toBeInTheDocument();
    expect(screen.getByText("Stalled")).toBeInTheDocument();
    expect(screen.getByText(/\$50,000/)).toBeInTheDocument();
    expect(screen.getByText("60%")).toBeInTheDocument();
  });

  it("renders relationship analysis", () => {
    const rawData: RawData = {
      relationship_analysis: [
        { opp: "Big Deal", engaged: 3, single_threaded: false },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Relationship Analysis (1)")).toBeInTheDocument();
    expect(screen.getByText("Big Deal")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument(); // boolean false
  });

  it("renders expand/save/reactivate categories", () => {
    const rawData: RawData = {
      expand: [{ name: "Growth Co", pipeline: 100000 }],
      save: [{ name: "At Risk Co", pipeline: 50000 }],
      reactivate: [{ name: "Dormant Co", last: "2023-01-01" }],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Expand (High Engagement) (1)")).toBeInTheDocument();
    expect(screen.getByText("Save (At Risk) (1)")).toBeInTheDocument();
    expect(screen.getByText("Re-activate (1)")).toBeInTheDocument();
  });

  it("skips _private keys and duckdb key", () => {
    const rawData: RawData = {
      today_meetings: [{ name: "Meeting" }],
      duckdb: "some query",
      _internal: [{ data: "hidden" }],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Today's Meetings (1)")).toBeInTheDocument();
    expect(screen.queryByText("duckdb")).not.toBeInTheDocument();
    expect(screen.queryByText("some query")).not.toBeInTheDocument();
    expect(screen.queryByText("_internal")).not.toBeInTheDocument();
  });

  it("shows +N more badge when more than 4 data types", () => {
    const rawData: RawData = {
      today_meetings: [{ name: "M1" }],
      at_risk_deals: [{ name: "D1" }],
      expand: [{ name: "E1" }],
      save: [{ name: "S1" }],
      reactivate: [{ name: "R1" }],
      slipped_deals: [{ name: "SD1" }],
    };

    render(<DataTables rawData={rawData} />);

    expect(screen.getByText("+2 more")).toBeInTheDocument();
  });

  // =========================================================================
  // GenericTable cell formatting
  // =========================================================================

  it("formats currency values in dynamic tables", () => {
    const rawData: RawData = {
      open_opportunities: [
        { name: "Deal", value: 75000, productTotal: 50000 },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText(/\$75,000/)).toBeInTheDocument();
    expect(screen.getByText(/\$50,000/)).toBeInTheDocument();
  });

  it("formats percentage values", () => {
    const rawData: RawData = {
      deals: [
        { name: "Deal", probability: 75, at_risk_pct: 10 },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText("10%")).toBeInTheDocument();
  });

  it("formats date strings", () => {
    const rawData: RawData = {
      deals: [
        { name: "Deal", closeDate: "2024-06-15" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    // formatDate should transform "2024-06-15" to readable format
    // Allow for timezone differences (Jun 14 or Jun 15)
    expect(screen.getByText(/Jun 1[45], 2024/)).toBeInTheDocument();
  });

  it("formats boolean values as Yes/No", () => {
    const rawData: RawData = {
      analysis: [
        { name: "Deal", single_threaded: true, active: false },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("formats arrays as item count", () => {
    const rawData: RawData = {
      meetings: [
        { name: "Meeting", contacts: [{ id: 1 }, { id: 2 }, { id: 3 }] },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("3 items")).toBeInTheDocument();
  });

  it("formats nested objects as [Object]", () => {
    const rawData: RawData = {
      deals: [
        { name: "Deal", primary_contact: { name: "John", email: "john@test.com" } },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("[Object]")).toBeInTheDocument();
  });

  it("truncates long text values in generic table", () => {
    const longText = "A".repeat(100);
    const rawData: RawData = {
      notes: [
        { id: "1", content: longText },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    const truncated = "A".repeat(80) + "…";
    expect(screen.getByText(truncated)).toBeInTheDocument();
  });

  it("hides columns ending with Id or ID", () => {
    const rawData: RawData = {
      items: [
        { title: "Item", companyId: "123", contactID: "456", visible: "yes" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Title")).toBeInTheDocument();
    expect(screen.getByText("Visible")).toBeInTheDocument();
    expect(screen.queryByText("companyId")).not.toBeInTheDocument();
    expect(screen.queryByText("contactID")).not.toBeInTheDocument();
  });

  it("returns null for generic table with empty data array", () => {
    const rawData: RawData = {
      empty_array: [],
    };

    const { container } = render(<DataTables rawData={rawData} />);

    expect(container.firstChild).toBeNull();
  });

  it("handles table when all columns are hidden", () => {
    const rawData: RawData = {
      items: [
        { _hidden: "value", id: "123", companyId: "456" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));
    // The header shows but no table content since all columns are hidden
    // The GenericTable returns null when no visible columns
    expect(screen.getByText(/Items \(1\)/)).toBeInTheDocument();
  });

  it("uses default icon for unknown entity types", () => {
    const rawData: RawData = {
      custom_data: [
        { name: "Custom Item" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    // Default icon is 📄
    expect(screen.getByText("📄")).toBeInTheDocument();
  });

  it("formats key with underscores to Title Case", () => {
    const rawData: RawData = {
      my_custom_entity: [
        { name: "Test" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("My Custom Entity (1)")).toBeInTheDocument();
  });

  it("shows empty string values as em dash", () => {
    const rawData: RawData = {
      items: [
        { name: "Test", description: "" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThan(0);
  });

  it("handles nested object without deals array", () => {
    const rawData: RawData = {
      summary: {
        total: 100,
        average: 50,
      },
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    // Should show scalar metrics from the nested object
    expect(screen.getByText("Summary Metrics")).toBeInTheDocument();
  });
});
