import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { DataExplorer } from "../components/DataExplorer";

// Mock fetch
const mockFetch = vi.fn();
(globalThis as typeof globalThis & { fetch: typeof fetch }).fetch = mockFetch;

// Sample test data
const mockCompaniesResponse = {
  data: [
    {
      company_id: "ACME-001",
      name: "Acme Corp",
      status: "Active",
      plan: "Pro",
      _private_texts: [
        {
          id: "note-1",
          text: "Important customer note",
          metadata_type: "note",
          metadata_created_at: "2025-01-01",
        },
      ],
    },
    {
      company_id: "BETA-002",
      name: "Beta Inc",
      status: "Trial",
      plan: "Standard",
      _private_texts: [],
    },
  ],
  total: 2,
  columns: ["company_id", "name", "status", "plan"],
};

const mockContactsResponse = {
  data: [
    {
      contact_id: "C-001",
      first_name: "John",
      last_name: "Doe",
      email: "john@acme.com",
      _private_texts: [],
    },
  ],
  total: 1,
  columns: ["contact_id", "first_name", "last_name", "email"],
};

const mockOpportunitiesResponse = {
  data: [
    {
      opportunity_id: "OPP-001",
      name: "Enterprise Deal",
      stage: "Proposal",
      value: "50000",
      _descriptions: [
        {
          title: "Deal Notes",
          text: "High priority customer",
          created_at: "2025-01-15",
        },
      ],
      _attachments: [
        {
          file_name: "proposal.pdf",
          file_size: "2MB",
        },
      ],
    },
  ],
  total: 1,
  columns: ["opportunity_id", "name", "stage", "value"],
};

const mockGroupsResponse = {
  data: [
    {
      group_id: "GRP-001",
      name: "VIP Customers",
      description: "High value accounts",
      _members: [
        {
          company_id: "ACME-001",
          added_at: "2025-01-01",
        },
      ],
    },
  ],
  total: 1,
  columns: ["group_id", "name", "description"],
};

const mockActivitiesResponse = {
  data: [
    {
      activity_id: "ACT-001",
      type: "Call",
      subject: "Follow-up call",
      company_id: "ACME-001",
    },
  ],
  total: 1,
  columns: ["activity_id", "type", "subject", "company_id"],
};

const mockHistoryResponse = {
  data: [
    {
      history_id: "HIST-001",
      action: "Email sent",
      company_id: "ACME-001",
      occurred_at: "2025-01-10",
    },
  ],
  total: 1,
  columns: ["history_id", "action", "company_id", "occurred_at"],
};

// Helper to mock fetch based on URL
function setupMockFetch() {
  mockFetch.mockImplementation((url: string) => {
    if (url.includes("/api/data/companies")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockCompaniesResponse),
      });
    }
    if (url.includes("/api/data/contacts")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockContactsResponse),
      });
    }
    if (url.includes("/api/data/opportunities")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockOpportunitiesResponse),
      });
    }
    if (url.includes("/api/data/groups")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockGroupsResponse),
      });
    }
    if (url.includes("/api/data/activities")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockActivitiesResponse),
      });
    }
    if (url.includes("/api/data/history")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockHistoryResponse),
      });
    }
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ data: [], total: 0, columns: [] }),
    });
  });
}

describe("DataExplorer", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    setupMockFetch();
  });

  describe("Rendering", () => {
    it("renders all tabs", async () => {
      render(<DataExplorer />);

      expect(screen.getByRole("tab", { name: /companies/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /contacts/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /opportunities/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /activities/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /groups/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /history/i })).toBeInTheDocument();
    });

    it("renders search input", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/search companies/i)).toBeInTheDocument();
      });
    });

    it("shows loading state initially", () => {
      mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves
      render(<DataExplorer />);

      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });

    it("shows error state on fetch failure", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText(/failed to load/i)).toBeInTheDocument();
      });
    });
  });

  describe("Data Display", () => {
    it("displays company data in table", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
        expect(screen.getByText("Beta Inc")).toBeInTheDocument();
      });
    });

    it("shows record count", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText(/2 of 2 records/i)).toBeInTheDocument();
      });
    });

    it("displays column headers", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Company Id")).toBeInTheDocument();
        expect(screen.getByText("Name")).toBeInTheDocument();
        expect(screen.getByText("Status")).toBeInTheDocument();
      });
    });
  });

  describe("Tab Navigation", () => {
    it("switches to contacts tab on click", async () => {
      render(<DataExplorer />);

      const contactsTab = screen.getByRole("tab", { name: /contacts/i });
      fireEvent.click(contactsTab);

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/search contacts/i)).toBeInTheDocument();
      });
    });

    it("switches to opportunities tab on click", async () => {
      render(<DataExplorer />);

      const oppTab = screen.getByRole("tab", { name: /opportunities/i });
      fireEvent.click(oppTab);

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/search opportunities/i)).toBeInTheDocument();
      });
    });

    it("switches to groups tab on click", async () => {
      render(<DataExplorer />);

      const groupsTab = screen.getByRole("tab", { name: /groups/i });
      fireEvent.click(groupsTab);

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/search groups/i)).toBeInTheDocument();
      });
    });

    it("marks active tab as selected", async () => {
      render(<DataExplorer />);

      const companiesTab = screen.getByRole("tab", { name: /companies/i });
      expect(companiesTab).toHaveAttribute("aria-selected", "true");

      const contactsTab = screen.getByRole("tab", { name: /contacts/i });
      expect(contactsTab).toHaveAttribute("aria-selected", "false");
    });
  });

  describe("Search Functionality", () => {
    it("filters data based on search term", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText(/search companies/i);
      fireEvent.change(searchInput, { target: { value: "Beta" } });

      await waitFor(() => {
        expect(screen.queryByText("Acme Corp")).not.toBeInTheDocument();
        expect(screen.getByText("Beta Inc")).toBeInTheDocument();
      });
    });

    it("shows filtered record count", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText(/2 of 2 records/i)).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText(/search companies/i);
      fireEvent.change(searchInput, { target: { value: "Beta" } });

      await waitFor(() => {
        expect(screen.getByText(/1 of 2 records/i)).toBeInTheDocument();
      });
    });

    it("clears search when switching tabs", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText(/search companies/i);
      fireEvent.change(searchInput, { target: { value: "test" } });

      const contactsTab = screen.getByRole("tab", { name: /contacts/i });
      fireEvent.click(contactsTab);

      await waitFor(() => {
        const newSearchInput = screen.getByPlaceholderText(/search contacts/i);
        expect(newSearchInput).toHaveValue("");
      });
    });
  });

  describe("Expandable Rows", () => {
    it("shows expand button for rows with nested data", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      // Acme Corp has _private_texts, so should have expand button
      const expandButtons = screen.getAllByRole("button", { name: /expand/i });
      expect(expandButtons.length).toBeGreaterThan(0);
    });

    it("expands row to show nested data on click", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const expandButton = screen.getAllByRole("button", { name: /expand/i })[0];
      fireEvent.click(expandButton);

      await waitFor(() => {
        expect(screen.getByText(/Notes & Attachments/i)).toBeInTheDocument();
        expect(screen.getByText(/Important customer note/i)).toBeInTheDocument();
      });
    });

    it("collapses expanded row on second click", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const expandButton = screen.getAllByRole("button", { name: /expand/i })[0];
      
      // Expand
      fireEvent.click(expandButton);
      await waitFor(() => {
        expect(screen.getByText(/Important customer note/i)).toBeInTheDocument();
      });

      // Collapse
      fireEvent.click(expandButton);
      await waitFor(() => {
        expect(screen.queryByText(/Important customer note/i)).not.toBeInTheDocument();
      });
    });
  });

  describe("Ask AI Button", () => {
    it("renders Ask AI buttons when onAskAbout is provided", async () => {
      const mockOnAskAbout = vi.fn();
      render(<DataExplorer onAskAbout={mockOnAskAbout} />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const askButtons = screen.getAllByTitle(/ask ai about this record/i);
      expect(askButtons.length).toBeGreaterThan(0);
    });

    it("calls onAskAbout with question when Ask AI button clicked", async () => {
      const mockOnAskAbout = vi.fn();
      render(<DataExplorer onAskAbout={mockOnAskAbout} />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const askButtons = screen.getAllByTitle(/ask ai about this record/i);
      fireEvent.click(askButtons[0]);

      expect(mockOnAskAbout).toHaveBeenCalledWith(
        expect.stringContaining("Acme Corp")
      );
    });

    it("does not render Ask AI buttons when onAskAbout is not provided", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const askButtons = screen.queryAllByTitle(/ask ai about this record/i);
      expect(askButtons.length).toBe(0);
    });
  });

  describe("Nested Data Types", () => {
    it("displays opportunity descriptions correctly", async () => {
      render(<DataExplorer />);

      const oppTab = screen.getByRole("tab", { name: /opportunities/i });
      fireEvent.click(oppTab);

      await waitFor(() => {
        expect(screen.getByText("Enterprise Deal")).toBeInTheDocument();
      });

      const expandButton = screen.getByRole("button", { name: /expand/i });
      fireEvent.click(expandButton);

      await waitFor(() => {
        expect(screen.getByText("Description")).toBeInTheDocument();
        expect(screen.getByText(/High priority customer/i)).toBeInTheDocument();
      });
    });

    it("displays opportunity attachments correctly", async () => {
      render(<DataExplorer />);

      const oppTab = screen.getByRole("tab", { name: /opportunities/i });
      fireEvent.click(oppTab);

      await waitFor(() => {
        expect(screen.getByText("Enterprise Deal")).toBeInTheDocument();
      });

      const expandButton = screen.getByRole("button", { name: /expand/i });
      fireEvent.click(expandButton);

      await waitFor(() => {
        expect(screen.getByText("Attachments")).toBeInTheDocument();
        expect(screen.getByText(/proposal.pdf/i)).toBeInTheDocument();
      });
    });

    it("displays group members correctly", async () => {
      render(<DataExplorer />);

      const groupsTab = screen.getByRole("tab", { name: /groups/i });
      fireEvent.click(groupsTab);

      await waitFor(() => {
        expect(screen.getByText("VIP Customers")).toBeInTheDocument();
      });

      const expandButton = screen.getByRole("button", { name: /expand/i });
      fireEvent.click(expandButton);

      await waitFor(() => {
        expect(screen.getByText("Members")).toBeInTheDocument();
        expect(screen.getByText(/ACME-001/i)).toBeInTheDocument();
      });
    });
  });

  describe("Empty States", () => {
    it("shows no records message when data is empty", async () => {
      mockFetch.mockImplementation(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ data: [], total: 0, columns: [] }),
        })
      );

      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText(/no records found/i)).toBeInTheDocument();
      });
    });

    it("shows no records when search returns no results", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText(/search companies/i);
      fireEvent.change(searchInput, { target: { value: "xyznonexistent" } });

      await waitFor(() => {
        expect(screen.getByText(/no records found/i)).toBeInTheDocument();
      });
    });
  });

  describe("Accessibility", () => {
    it("has proper ARIA roles for tabs", async () => {
      render(<DataExplorer />);

      expect(screen.getByRole("tablist")).toBeInTheDocument();
      expect(screen.getAllByRole("tab")).toHaveLength(6);
      expect(screen.getByRole("tabpanel")).toBeInTheDocument();
    });

    it("search input has proper label", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        const searchInput = screen.getByPlaceholderText(/search companies/i);
        expect(searchInput).toHaveAttribute("aria-label", "Search Companies");
      });
    });

    it("expand buttons have proper aria-expanded state", async () => {
      render(<DataExplorer />);

      await waitFor(() => {
        expect(screen.getByText("Acme Corp")).toBeInTheDocument();
      });

      const expandButton = screen.getAllByRole("button", { name: /expand/i })[0];
      expect(expandButton).toHaveAttribute("aria-expanded", "false");

      fireEvent.click(expandButton);

      await waitFor(() => {
        expect(expandButton).toHaveAttribute("aria-expanded", "true");
      });
    });
  });
});
