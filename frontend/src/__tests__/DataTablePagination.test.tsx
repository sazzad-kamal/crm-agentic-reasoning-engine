import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { DataTable } from "../components/dataExplorer/DataTable";

// Generate test data
const generateTestData = (count: number): Record<string, unknown>[] => {
  return Array.from({ length: count }, (_, i) => ({
    id: `ID-${i + 1}`,
    name: `Company ${i + 1}`,
    status: i % 2 === 0 ? "Active" : "Inactive",
  }));
};

describe("DataTable Pagination", () => {
  const columns = ["id", "name", "status"];

  // ===========================================================================
  // Basic Pagination Rendering
  // ===========================================================================

  describe("basic rendering", () => {
    it("does not show pagination for small datasets", () => {
      const data = generateTestData(5);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      expect(screen.queryByRole("navigation", { name: /pagination/i })).not.toBeInTheDocument();
    });

    it("shows pagination for large datasets", () => {
      const data = generateTestData(15);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      expect(screen.getByRole("navigation", { name: /pagination/i })).toBeInTheDocument();
    });

    it("shows correct number of pages", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      // Should show 3 pages (25 items / 10 per page = 3 pages)
      expect(screen.getByRole("button", { name: /go to page 1/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /go to page 2/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /go to page 3/i })).toBeInTheDocument();
    });

    it("shows item range correctly", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      expect(screen.getByText(/showing 1–10 of 25/i)).toBeInTheDocument();
    });
  });

  // ===========================================================================
  // Navigation
  // ===========================================================================

  describe("navigation", () => {
    it("navigates to next page", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      expect(screen.getByText("Company 1")).toBeInTheDocument();
      expect(screen.queryByText("Company 11")).not.toBeInTheDocument();

      const nextBtn = screen.getByRole("button", { name: /next/i });
      fireEvent.click(nextBtn);

      expect(screen.queryByText("Company 1")).not.toBeInTheDocument();
      expect(screen.getByText("Company 11")).toBeInTheDocument();
      expect(screen.getByText(/showing 11–20 of 25/i)).toBeInTheDocument();
    });

    it("navigates to previous page", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      // Go to page 2
      fireEvent.click(screen.getByRole("button", { name: /next/i }));
      expect(screen.getByText("Company 11")).toBeInTheDocument();

      // Go back to page 1
      const prevBtn = screen.getByRole("button", { name: /prev/i });
      fireEvent.click(prevBtn);

      expect(screen.getByText("Company 1")).toBeInTheDocument();
      expect(screen.queryByText("Company 11")).not.toBeInTheDocument();
    });

    it("navigates to specific page", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      const page3Btn = screen.getByRole("button", { name: /go to page 3/i });
      fireEvent.click(page3Btn);

      expect(screen.getByText("Company 21")).toBeInTheDocument();
      expect(screen.getByText(/showing 21–25 of 25/i)).toBeInTheDocument();
    });

    it("disables previous button on first page", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      const prevBtn = screen.getByRole("button", { name: /prev/i });
      expect(prevBtn).toBeDisabled();
    });

    it("disables next button on last page", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      // Navigate to last page
      fireEvent.click(screen.getByRole("button", { name: /go to page 3/i }));

      const nextBtn = screen.getByRole("button", { name: /next/i });
      expect(nextBtn).toBeDisabled();
    });
  });

  // ===========================================================================
  // Page Size
  // ===========================================================================

  describe("page size", () => {
    it("respects custom page size", () => {
      const data = generateTestData(15);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={5} />);

      // Should show 3 pages with pageSize=5 (15 items / 5 per page = 3 pages)
      expect(screen.getByRole("button", { name: /go to page 3/i })).toBeInTheDocument();
      expect(screen.getByText(/showing 1–5 of 15/i)).toBeInTheDocument();
    });

    it("uses default page size of 10", () => {
      const data = generateTestData(15);
      render(<DataTable data={data} columns={columns} dataType="companies" />);

      // Default pageSize=10 means 2 pages for 15 items
      expect(screen.getByRole("button", { name: /go to page 2/i })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /go to page 3/i })).not.toBeInTheDocument();
    });
  });

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe("edge cases", () => {
    it("handles empty data", () => {
      render(<DataTable data={[]} columns={columns} dataType="companies" pageSize={10} />);

      expect(screen.getByText(/no records found/i)).toBeInTheDocument();
      expect(screen.queryByRole("navigation", { name: /pagination/i })).not.toBeInTheDocument();
    });

    it("handles exactly one page of data", () => {
      const data = generateTestData(10);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      expect(screen.queryByRole("navigation", { name: /pagination/i })).not.toBeInTheDocument();
    });

    it("shows last page correctly with partial data", () => {
      const data = generateTestData(23);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      // Navigate to last page
      fireEvent.click(screen.getByRole("button", { name: /go to page 3/i }));

      expect(screen.getByText(/showing 21–23 of 23/i)).toBeInTheDocument();
      expect(screen.getByText("Company 21")).toBeInTheDocument();
      expect(screen.getByText("Company 22")).toBeInTheDocument();
      expect(screen.getByText("Company 23")).toBeInTheDocument();
    });

    it("clamps to valid page range when data changes", () => {
      const initialData = generateTestData(50);
      const { rerender } = render(
        <DataTable data={initialData} columns={columns} dataType="companies" pageSize={10} />
      );

      // Navigate to page 5
      fireEvent.click(screen.getByRole("button", { name: /go to page 5/i }));
      expect(screen.getByText("Company 41")).toBeInTheDocument();

      // Reduce data to only 20 items (2 pages)
      const reducedData = generateTestData(20);
      rerender(<DataTable data={reducedData} columns={columns} dataType="companies" pageSize={10} />);

      // Should clamp to page 2 (last valid page)
      expect(screen.getByText("Company 11")).toBeInTheDocument();
      expect(screen.getByText(/showing 11–20 of 20/i)).toBeInTheDocument();
    });
  });

  // ===========================================================================
  // Ellipsis
  // ===========================================================================

  describe("ellipsis for many pages", () => {
    it("shows ellipsis for many pages", () => {
      const data = generateTestData(100);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      // Get the pagination navigation specifically
      const nav = screen.getByRole("navigation", { name: /pagination/i });

      // Should show page 1, some pages, ellipsis, and page 10
      // Use exact match to avoid "page 1" matching "page 10"
      expect(within(nav).getByRole("button", { name: "Go to page 1" })).toBeInTheDocument();
      expect(within(nav).getByRole("button", { name: "Go to page 10" })).toBeInTheDocument();
      expect(within(nav).getByText("...")).toBeInTheDocument();
    });

    it("updates ellipsis position when navigating to middle", () => {
      const data = generateTestData(100);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      // Navigate to page 2, then page 3, to get closer to middle
      fireEvent.click(screen.getByRole("button", { name: /go to page 2/i }));
      // After clicking page 2, page 3 should be visible
      fireEvent.click(screen.getByRole("button", { name: /go to page 3/i }));
      // After clicking page 3, page 4 should be visible
      fireEvent.click(screen.getByRole("button", { name: /go to page 4/i }));
      // Now we're at page 4, and page 5 should be visible
      fireEvent.click(screen.getByRole("button", { name: /go to page 5/i }));

      // At page 5, should have ellipsis on both sides
      const ellipses = screen.getAllByText("...");
      expect(ellipses.length).toBe(2);
    });
  });

  // ===========================================================================
  // Expanded Rows
  // ===========================================================================

  describe("expanded rows and pagination", () => {
    it("collapses expanded rows when changing pages", () => {
      const dataWithNested = Array.from({ length: 15 }, (_, i) => ({
        id: `ID-${i + 1}`,
        name: `Company ${i + 1}`,
        _private_texts: [{ id: `note-${i}`, text: `NestedContent${i + 1}` }],
      }));
      const nestedFields = [{ key: "_private_texts", label: "Notes" }];

      render(
        <DataTable
          data={dataWithNested}
          columns={["id", "name"]}
          dataType="companies"
          pageSize={10}
          nestedFields={nestedFields}
        />
      );

      // Expand first row
      const expandButtons = screen.getAllByRole("button", { name: /expand/i });
      fireEvent.click(expandButtons[0]);

      // First row's nested content should be visible
      expect(screen.getByText(/NestedContent1/)).toBeInTheDocument();

      // Navigate to next page
      fireEvent.click(screen.getByRole("button", { name: /next/i }));

      // Expanded content from page 1 should not be visible
      expect(screen.queryByText(/NestedContent1/)).not.toBeInTheDocument();
    });
  });

  // ===========================================================================
  // Accessibility
  // ===========================================================================

  describe("accessibility", () => {
    it("has proper ARIA label for navigation", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      const nav = screen.getByRole("navigation");
      expect(nav).toHaveAttribute("aria-label", expect.stringContaining("Pagination"));
    });

    it("has aria-current on active page", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      const page1Btn = screen.getByRole("button", { name: /go to page 1/i });
      expect(page1Btn).toHaveAttribute("aria-current", "page");

      const page2Btn = screen.getByRole("button", { name: /go to page 2/i });
      expect(page2Btn).not.toHaveAttribute("aria-current");
    });

    it("updates aria-current when navigating", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      fireEvent.click(screen.getByRole("button", { name: /go to page 2/i }));

      const page1Btn = screen.getByRole("button", { name: /go to page 1/i });
      expect(page1Btn).not.toHaveAttribute("aria-current");

      const page2Btn = screen.getByRole("button", { name: /go to page 2/i });
      expect(page2Btn).toHaveAttribute("aria-current", "page");
    });

    it("has aria-live on page info", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      const info = screen.getByText(/showing 1–10 of 25/i);
      expect(info).toHaveAttribute("aria-live", "polite");
    });

    it("next/prev buttons have descriptive aria-labels", () => {
      const data = generateTestData(25);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      expect(screen.getByRole("button", { name: /previous page/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /next page/i })).toBeInTheDocument();
    });

    it("hides ellipsis from screen readers", () => {
      const data = generateTestData(100);
      render(<DataTable data={data} columns={columns} dataType="companies" pageSize={10} />);

      const ellipsis = screen.getByText("...");
      expect(ellipsis).toHaveAttribute("aria-hidden", "true");
    });
  });

  // ===========================================================================
  // Ask AI with Pagination
  // ===========================================================================

  describe("ask AI with pagination", () => {
    it("ask AI works correctly with paginated data", () => {
      const mockOnAskAbout = vi.fn();
      const data = generateTestData(25);

      render(
        <DataTable
          data={data}
          columns={columns}
          dataType="companies"
          pageSize={10}
          onAskAbout={mockOnAskAbout}
        />
      );

      // Navigate to page 2
      fireEvent.click(screen.getByRole("button", { name: /next/i }));

      // Click Ask AI on first visible row (Company 11)
      const askButtons = screen.getAllByTitle(/ask ai about this record/i);
      fireEvent.click(askButtons[0]);

      expect(mockOnAskAbout).toHaveBeenCalledWith(
        expect.stringContaining("Company 11")
      );
    });
  });
});
