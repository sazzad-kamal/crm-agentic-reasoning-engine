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
    expect(screen.getByText("Data used")).toBeInTheDocument();
  });

  // =========================================================================
  // Nested Items - Private Texts
  // =========================================================================

  describe("NestedItems - private_texts", () => {
    it("renders private texts in companies table", () => {
      const rawData: RawData = {
        companies: [
          {
            company_id: "1",
            name: "Acme Corp",
            plan: "Pro",
            renewal_date: "2024-12-31",
            _private_texts: [
              { id: "1", type: "note", title: "Important meeting notes", text: "Details here" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.getByText("note")).toBeInTheDocument();
      expect(screen.getByText("Important meeting notes")).toBeInTheDocument();
    });

    it("shows type badge for private text", () => {
      const rawData: RawData = {
        companies: [
          {
            company_id: "1",
            name: "Test",
            plan: "Pro",
            renewal_date: "2024-12-31",
            _private_texts: [
              { id: "1", type: "history", title: "History entry", text: "" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.getByText("history")).toBeInTheDocument();
    });

    it("falls back to text when title is empty", () => {
      const rawData: RawData = {
        companies: [
          {
            company_id: "1",
            name: "Test",
            plan: "Pro",
            renewal_date: "2024-12-31",
            _private_texts: [
              { id: "1", type: "note", title: "", text: "Fallback text content" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.getByText("Fallback text content")).toBeInTheDocument();
    });

    it("truncates long text to 100 characters", () => {
      const longText = "A".repeat(150);
      const rawData: RawData = {
        companies: [
          {
            company_id: "1",
            name: "Test",
            plan: "Pro",
            renewal_date: "2024-12-31",
            _private_texts: [
              { id: "1", type: "note", title: longText, text: "" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      // Should show truncated text (100 chars)
      expect(screen.getByText("A".repeat(100))).toBeInTheDocument();
    });

    it("shows +X more when more than 5 items", () => {
      const rawData: RawData = {
        companies: [
          {
            company_id: "1",
            name: "Test",
            plan: "Pro",
            renewal_date: "2024-12-31",
            _private_texts: [
              { id: "1", type: "note", title: "Note 1", text: "" },
              { id: "2", type: "note", title: "Note 2", text: "" },
              { id: "3", type: "note", title: "Note 3", text: "" },
              { id: "4", type: "note", title: "Note 4", text: "" },
              { id: "5", type: "note", title: "Note 5", text: "" },
              { id: "6", type: "note", title: "Note 6", text: "" },
              { id: "7", type: "note", title: "Note 7", text: "" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.getByText("+2 more")).toBeInTheDocument();
      expect(screen.getByText("Note 5")).toBeInTheDocument();
      expect(screen.queryByText("Note 6")).not.toBeInTheDocument();
    });

    it("renders nothing for empty private_texts", () => {
      const rawData: RawData = {
        companies: [
          {
            company_id: "1",
            name: "Test",
            plan: "Pro",
            renewal_date: "2024-12-31",
            _private_texts: [],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.queryByText("note")).not.toBeInTheDocument();
    });
  });

  // =========================================================================
  // Nested Items - Attachments
  // =========================================================================

  describe("NestedItems - attachments", () => {
    it("renders attachments in opportunities table", () => {
      const rawData: RawData = {
        opportunities: [
          {
            opportunity_id: "1",
            name: "Big Deal",
            stage: "Negotiation",
            expected_close_date: "2024-06-30",
            value: 100000,
            _attachments: [
              { attachment_id: "1", title: "Proposal.pdf", file_type: "pdf", summary: "", created_at: "" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.getByText("📎")).toBeInTheDocument();
      expect(screen.getByText("Proposal.pdf")).toBeInTheDocument();
      expect(screen.getByText("pdf")).toBeInTheDocument();
    });

    it("falls back to file_name when title is empty", () => {
      const rawData: RawData = {
        opportunities: [
          {
            opportunity_id: "1",
            name: "Deal",
            stage: "Open",
            expected_close_date: "2024-06-30",
            value: 50000,
            _attachments: [
              { attachment_id: "1", title: "", file_name: "contract.docx", file_type: "docx", summary: "", created_at: "" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.getByText("contract.docx")).toBeInTheDocument();
    });

    it("shows default Attachment when no title or file_name", () => {
      const rawData: RawData = {
        opportunities: [
          {
            opportunity_id: "1",
            name: "Deal",
            stage: "Open",
            expected_close_date: "2024-06-30",
            value: 50000,
            _attachments: [
              { attachment_id: "1", title: "", file_type: "", summary: "", created_at: "" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.getByText("Attachment")).toBeInTheDocument();
    });

    it("shows +X more for attachments when more than 5", () => {
      const rawData: RawData = {
        opportunities: [
          {
            opportunity_id: "1",
            name: "Deal",
            stage: "Open",
            expected_close_date: "2024-06-30",
            value: 50000,
            _attachments: [
              { attachment_id: "1", title: "File 1", file_type: "pdf", summary: "", created_at: "" },
              { attachment_id: "2", title: "File 2", file_type: "pdf", summary: "", created_at: "" },
              { attachment_id: "3", title: "File 3", file_type: "pdf", summary: "", created_at: "" },
              { attachment_id: "4", title: "File 4", file_type: "pdf", summary: "", created_at: "" },
              { attachment_id: "5", title: "File 5", file_type: "pdf", summary: "", created_at: "" },
              { attachment_id: "6", title: "File 6", file_type: "pdf", summary: "", created_at: "" },
              { attachment_id: "7", title: "File 7", file_type: "pdf", summary: "", created_at: "" },
              { attachment_id: "8", title: "File 8", file_type: "pdf", summary: "", created_at: "" },
            ],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.getByText("+3 more")).toBeInTheDocument();
      expect(screen.getByText("File 5")).toBeInTheDocument();
      expect(screen.queryByText("File 6")).not.toBeInTheDocument();
    });

    it("renders nothing for empty attachments", () => {
      const rawData: RawData = {
        opportunities: [
          {
            opportunity_id: "1",
            name: "Deal",
            stage: "Open",
            expected_close_date: "2024-06-30",
            value: 50000,
            _attachments: [],
          },
        ],
      };

      render(<DataTables rawData={rawData} />);
      fireEvent.click(screen.getByRole("button"));

      expect(screen.queryByText("📎")).not.toBeInTheDocument();
    });
  });

  // =========================================================================
  // Table Headers with Nested Columns
  // =========================================================================

  it("companies table has Notes header for nested data", () => {
    const rawData: RawData = {
      companies: [
        { company_id: "1", name: "Test", plan: "Basic", renewal_date: "2024-12-31" },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Notes")).toBeInTheDocument();
  });

  it("opportunities table has Attachments header", () => {
    const rawData: RawData = {
      opportunities: [
        {
          opportunity_id: "1",
          name: "Deal",
          stage: "Open",
          expected_close_date: "2024-06-30",
          value: 50000,
        },
      ],
    };

    render(<DataTables rawData={rawData} />);
    fireEvent.click(screen.getByRole("button"));

    expect(screen.getByText("Attachments")).toBeInTheDocument();
  });
});
