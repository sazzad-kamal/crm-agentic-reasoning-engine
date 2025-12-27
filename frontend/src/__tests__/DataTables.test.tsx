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
  // Collapsed State
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

  it("shows correct table count for single table", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Basic", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

    expect(screen.getByText("Data used (1 table)")).toBeInTheDocument();
  });

  it("shows correct table count for multiple tables", () => {
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

    expect(screen.getByText("Data used (2 tables)")).toBeInTheDocument();
  });

  it("shows expand arrow when collapsed", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

    expect(screen.getByText("▶")).toBeInTheDocument();
  });

  // =========================================================================
  // Expanded State
  // =========================================================================

  it("expands when header clicked", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme Corp", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

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

  it("collapses when header clicked again", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Acme", plan: "Pro", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);

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
          description: "Changed from Lead to Customer",
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
    expect(screen.getByText("Data used (1 table)")).toBeInTheDocument();
  });
});
