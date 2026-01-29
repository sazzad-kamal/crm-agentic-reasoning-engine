import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DataTable } from "../components/dataExplorer/DataTable";

describe("DataTable sorting and coverage", () => {
  const columns = ["id", "name", "value"];

  const data = [
    { id: "1", name: "Beta Corp", value: 200 },
    { id: "2", name: "Alpha Inc", value: 100 },
    { id: "3", name: "Gamma LLC", value: 300 },
    { id: "4", name: "Delta Co", value: null },
  ];

  // ===========================================================================
  // Sorting
  // ===========================================================================

  it("sorts ascending on first click", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const nameHeader = screen.getByText("Name");
    fireEvent.click(nameHeader);

    const rows = screen.getAllByRole("row");
    // First row is header, so data rows start at index 1
    expect(rows[1]).toHaveTextContent("Alpha Inc");
    expect(rows[2]).toHaveTextContent("Beta Corp");
  });

  it("sorts descending on second click", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const nameHeader = screen.getByText("Name");
    fireEvent.click(nameHeader); // asc
    fireEvent.click(nameHeader); // desc

    const rows = screen.getAllByRole("row");
    // Nulls go to end in desc, so Delta (null value) isn't at top for name sort
    expect(rows[1]).toHaveTextContent("Gamma LLC");
    expect(rows[2]).toHaveTextContent("Delta Co");
  });

  it("clears sort on third click", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const nameHeader = screen.getByText("Name");
    fireEvent.click(nameHeader); // asc
    fireEvent.click(nameHeader); // desc
    fireEvent.click(nameHeader); // clear

    // Should return to original order
    const rows = screen.getAllByRole("row");
    expect(rows[1]).toHaveTextContent("Beta Corp");
  });

  it("sorts numeric columns correctly", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const valueHeader = screen.getByText("Value");
    fireEvent.click(valueHeader); // asc

    const rows = screen.getAllByRole("row");
    expect(rows[1]).toHaveTextContent("100"); // Alpha has lowest value
  });

  it("handles null values in sort (nulls go to end in asc)", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const valueHeader = screen.getByText("Value");
    fireEvent.click(valueHeader); // asc - nulls at end

    const rows = screen.getAllByRole("row");
    expect(rows[4]).toHaveTextContent("Delta Co"); // null value at end
  });

  it("handles null values in sort (nulls go to start in desc)", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const valueHeader = screen.getByText("Value");
    fireEvent.click(valueHeader); // asc
    fireEvent.click(valueHeader); // desc - nulls at start

    const rows = screen.getAllByRole("row");
    expect(rows[1]).toHaveTextContent("Delta Co"); // null value at start
  });

  it("switches sort column", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    // Sort by name
    fireEvent.click(screen.getByText("Name"));
    // Switch to sort by value
    fireEvent.click(screen.getByText("Value"));

    const valueHeader = screen.getByText("Value").closest("th");
    expect(valueHeader).toHaveAttribute("aria-sort", "ascending");
  });

  it("shows sort indicators", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const nameHeader = screen.getByText("Name");
    fireEvent.click(nameHeader); // asc

    const th = nameHeader.closest("th");
    expect(th).toHaveClass("data-table__th--sorted");
    expect(th).toHaveAttribute("aria-sort", "ascending");
  });

  it("resets to page 1 when sorting", () => {
    // Need enough data for pagination
    const largeData = Array.from({ length: 25 }, (_, i) => ({
      id: `${i + 1}`,
      name: `Company ${i + 1}`,
      value: i * 10,
    }));

    render(<DataTable data={largeData} columns={columns} dataType="companies" pageSize={10} />);

    // Go to page 2
    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(screen.getByText(/showing 11–20/i)).toBeInTheDocument();

    // Sort by name
    fireEvent.click(screen.getByText("Name"));

    // Should be back on page 1
    expect(screen.getByText(/showing 1–10/i)).toBeInTheDocument();
  });

  // ===========================================================================
  // Keyboard Navigation for Sort
  // ===========================================================================

  it("sorts on Enter key", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const nameHeader = screen.getByText("Name").closest("th")!;
    fireEvent.keyDown(nameHeader, { key: "Enter" });

    expect(nameHeader).toHaveAttribute("aria-sort", "ascending");
  });

  it("sorts on Space key", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const nameHeader = screen.getByText("Name").closest("th")!;
    fireEvent.keyDown(nameHeader, { key: " " });

    expect(nameHeader).toHaveAttribute("aria-sort", "ascending");
  });

  it("does not sort on other keys", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const nameHeader = screen.getByText("Name").closest("th")!;
    fireEvent.keyDown(nameHeader, { key: "a" });

    expect(nameHeader).toHaveAttribute("aria-sort", "none");
  });

  // ===========================================================================
  // formatValue edge cases
  // ===========================================================================

  it("shows dash for null values", () => {
    const dataWithNull = [{ id: "1", name: null, value: null }];
    render(<DataTable data={dataWithNull} columns={columns} dataType="companies" />);

    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(2);
  });

  it("shows dash for undefined values", () => {
    const dataWithUndefined = [{ id: "1", name: undefined, value: undefined }];
    render(<DataTable data={dataWithUndefined as unknown as Record<string, unknown>[]} columns={columns} dataType="companies" />);

    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(2);
  });

  it("truncates long values", () => {
    const longValue = "A".repeat(100);
    const dataWithLong = [{ id: "1", name: longValue, value: 0 }];
    render(<DataTable data={dataWithLong} columns={columns} dataType="companies" />);

    // Should be truncated to 50 chars + "..."
    expect(screen.getByText("A".repeat(50) + "...")).toBeInTheDocument();
  });

  // ===========================================================================
  // getQuestion for different data types
  // ===========================================================================

  it("generates correct question for contacts", () => {
    const mockOnAskAbout = vi.fn();
    const contactData = [{ first_name: "John", last_name: "Doe", company_id: "Acme" }];
    render(
      <DataTable
        data={contactData}
        columns={["first_name", "last_name", "company_id"]}
        dataType="contacts"
        onAskAbout={mockOnAskAbout}
      />
    );

    const askBtn = screen.getByTitle(/ask ai about this record/i);
    fireEvent.click(askBtn);

    expect(mockOnAskAbout).toHaveBeenCalledWith("Tell me about John Doe at Acme");
  });

  it("generates correct question for opportunities", () => {
    const mockOnAskAbout = vi.fn();
    const oppData = [{ name: "Big Deal", stage: "Negotiation" }];
    render(
      <DataTable
        data={oppData}
        columns={["name", "stage"]}
        dataType="opportunities"
        onAskAbout={mockOnAskAbout}
      />
    );

    const askBtn = screen.getByTitle(/ask ai about this record/i);
    fireEvent.click(askBtn);

    expect(mockOnAskAbout).toHaveBeenCalledWith("What's the status of the Big Deal opportunity?");
  });

  it("generates correct question for activities", () => {
    const mockOnAskAbout = vi.fn();
    const actData = [{ company_id: "TechCorp", type: "call" }];
    render(
      <DataTable
        data={actData}
        columns={["company_id", "type"]}
        dataType="activities"
        onAskAbout={mockOnAskAbout}
      />
    );

    const askBtn = screen.getByTitle(/ask ai about this record/i);
    fireEvent.click(askBtn);

    expect(mockOnAskAbout).toHaveBeenCalledWith("What activities are scheduled for TechCorp?");
  });

  it("generates correct question for history", () => {
    const mockOnAskAbout = vi.fn();
    const histData = [{ company_id: "Acme", event_type: "upgrade" }];
    render(
      <DataTable
        data={histData}
        columns={["company_id", "event_type"]}
        dataType="history"
        onAskAbout={mockOnAskAbout}
      />
    );

    const askBtn = screen.getByTitle(/ask ai about this record/i);
    fireEvent.click(askBtn);

    expect(mockOnAskAbout).toHaveBeenCalledWith("Show me the history for Acme");
  });

  it("generates fallback question for unknown data types", () => {
    const mockOnAskAbout = vi.fn();
    const unknownData = [{ field1: "Value1" }];
    render(
      <DataTable
        data={unknownData}
        columns={["field1"]}
        dataType={"unknown" as never}
        onAskAbout={mockOnAskAbout}
      />
    );

    const askBtn = screen.getByTitle(/ask ai about this record/i);
    fireEvent.click(askBtn);

    expect(mockOnAskAbout).toHaveBeenCalledWith("Tell me about Value1");
  });

  // ===========================================================================
  // Column filtering (_private fields)
  // ===========================================================================

  it("filters out columns starting with underscore", () => {
    const dataWithPrivate = [{ id: "1", name: "Test", _private_texts: [] }];
    render(
      <DataTable
        data={dataWithPrivate}
        columns={["id", "name", "_private_texts"]}
        dataType="companies"
      />
    );

    // _private_texts column should not appear in headers
    expect(screen.queryByText("Private Texts")).not.toBeInTheDocument();
    expect(screen.getByText("Id")).toBeInTheDocument();
    expect(screen.getByText("Name")).toBeInTheDocument();
  });

  // ===========================================================================
  // Expandable rows
  // ===========================================================================

  it("shows expand button for rows with nested data", () => {
    const dataWithNested = [
      { id: "1", name: "Test", _private_texts: [{ id: "n1", text: "Note content" }] },
    ];
    const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

    render(
      <DataTable
        data={dataWithNested}
        columns={["id", "name"]}
        dataType="companies"
        nestedFields={nestedFields}
      />
    );

    expect(screen.getByRole("button", { name: /expand details/i })).toBeInTheDocument();
  });

  it("does not show expand button for rows without nested data", () => {
    const dataWithoutNested = [{ id: "1", name: "Test", _private_texts: [] }];
    const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

    render(
      <DataTable
        data={dataWithoutNested}
        columns={["id", "name"]}
        dataType="companies"
        nestedFields={nestedFields}
      />
    );

    expect(screen.queryByRole("button", { name: /expand details/i })).not.toBeInTheDocument();
  });

  it("toggles expanded state on click", () => {
    const dataWithNested = [
      { id: "1", name: "Test", _private_texts: [{ id: "n1", text: "Note content" }] },
    ];
    const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

    render(
      <DataTable
        data={dataWithNested}
        columns={["id", "name"]}
        dataType="companies"
        nestedFields={nestedFields}
      />
    );

    const expandBtn = screen.getByRole("button", { name: /expand details/i });
    fireEvent.click(expandBtn);

    // Should now show collapse
    expect(screen.getByRole("button", { name: /collapse details/i })).toBeInTheDocument();

    // Click again to collapse
    fireEvent.click(screen.getByRole("button", { name: /collapse details/i }));
    expect(screen.getByRole("button", { name: /expand details/i })).toBeInTheDocument();
  });

  // ===========================================================================
  // Table accessibility
  // ===========================================================================

  it("has visually hidden caption", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const caption = screen.getByText(/companies data with 4 records/i);
    expect(caption).toBeInTheDocument();
    expect(caption).toHaveClass("visually-hidden");
  });

  it("has scrollable region with role", () => {
    const { container } = render(<DataTable data={data} columns={columns} dataType="companies" />);

    const region = container.querySelector('[role="region"]');
    expect(region).toHaveAttribute("aria-label", "companies data table");
    expect(region).toHaveAttribute("tabIndex", "0");
  });

  it("column headers are sortable", () => {
    render(<DataTable data={data} columns={columns} dataType="companies" />);

    const headers = screen.getAllByRole("columnheader");
    headers.forEach((header) => {
      expect(header).toHaveAttribute("tabIndex", "0");
      expect(header).toHaveAttribute("aria-sort");
    });
  });

  it("rows with both null values sort equal", () => {
    const dataWithBothNull = [
      { id: "1", name: null, value: null },
      { id: "2", name: null, value: null },
    ];
    render(<DataTable data={dataWithBothNull} columns={columns} dataType="companies" />);

    fireEvent.click(screen.getByText("Value"));

    // Both should render without errors
    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(3); // header + 2 data rows
  });
});
