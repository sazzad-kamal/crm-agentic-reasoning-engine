import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SourceChip, SourcesRow } from "../components/SourceChip";
import type { Source } from "../types";

describe("SourceChip", () => {
  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders company source with correct icon and label", () => {
    const source: Source = {
      id: "comp1",
      type: "company",
      label: "Acme Corp",
    };

    render(<SourceChip source={source} />);

    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
    expect(screen.getByText("🏢")).toBeInTheDocument();
  });

  it("renders doc source with correct icon and label", () => {
    const source: Source = {
      id: "doc1",
      type: "doc",
      label: "User Guide",
    };

    render(<SourceChip source={source} />);

    expect(screen.getByText("User Guide")).toBeInTheDocument();
    expect(screen.getByText("📄")).toBeInTheDocument();
  });

  it("renders activity source with correct icon", () => {
    const source: Source = {
      id: "act1",
      type: "activity",
      label: "Meeting notes",
    };

    render(<SourceChip source={source} />);

    expect(screen.getByText("Meeting notes")).toBeInTheDocument();
    expect(screen.getByText("📋")).toBeInTheDocument();
  });

  it("renders opportunity source with correct icon", () => {
    const source: Source = {
      id: "opp1",
      type: "opportunity",
      label: "Q1 Deal",
    };

    render(<SourceChip source={source} />);

    expect(screen.getByText("Q1 Deal")).toBeInTheDocument();
    expect(screen.getByText("💰")).toBeInTheDocument();
  });

  it("renders history source with correct icon", () => {
    const source: Source = {
      id: "hist1",
      type: "history",
      label: "Account timeline",
    };

    render(<SourceChip source={source} />);

    expect(screen.getByText("Account timeline")).toBeInTheDocument();
    expect(screen.getByText("📜")).toBeInTheDocument();
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("has proper ARIA role", () => {
    const source: Source = {
      id: "comp1",
      type: "company",
      label: "Acme Corp",
    };

    render(<SourceChip source={source} />);

    const chip = screen.getByRole("listitem");
    expect(chip).toBeInTheDocument();
  });

  it("has descriptive ARIA label", () => {
    const source: Source = {
      id: "comp1",
      type: "company",
      label: "Acme Corp",
    };

    render(<SourceChip source={source} />);

    expect(screen.getByLabelText("Company: Acme Corp")).toBeInTheDocument();
  });

  it("hides icon from screen readers", () => {
    const source: Source = {
      id: "comp1",
      type: "company",
      label: "Acme Corp",
    };

    const { container } = render(<SourceChip source={source} />);
    const icon = container.querySelector(".source-chip__icon");

    expect(icon).toHaveAttribute("aria-hidden", "true");
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles unknown source type with fallback icon", () => {
    const source = {
      id: "unknown1",
      type: "custom" as Source["type"],
      label: "Custom source",
    };

    render(<SourceChip source={source} />);

    expect(screen.getByText("Custom source")).toBeInTheDocument();
    expect(screen.getByText("📌")).toBeInTheDocument();
  });

  it("handles empty label", () => {
    const source: Source = {
      id: "comp1",
      type: "company",
      label: "",
    };

    render(<SourceChip source={source} />);

    const chip = screen.getByRole("listitem");
    expect(chip).toBeInTheDocument();
  });

  it("handles very long label", () => {
    const source: Source = {
      id: "comp1",
      type: "company",
      label: "A".repeat(200),
    };

    render(<SourceChip source={source} />);

    expect(screen.getByText("A".repeat(200))).toBeInTheDocument();
  });

  it("handles special characters in label", () => {
    const source: Source = {
      id: "comp1",
      type: "company",
      label: "Company & Co. <Inc>",
    };

    render(<SourceChip source={source} />);

    expect(screen.getByText("Company & Co. <Inc>")).toBeInTheDocument();
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("re-renders when source changes", () => {
    const source1: Source = {
      id: "comp1",
      type: "company",
      label: "Acme Corp",
    };

    const source2: Source = {
      id: "comp2",
      type: "doc",
      label: "User Guide",
    };

    const { rerender } = render(<SourceChip source={source1} />);
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();

    rerender(<SourceChip source={source2} />);
    expect(screen.queryByText("Acme Corp")).not.toBeInTheDocument();
    expect(screen.getByText("User Guide")).toBeInTheDocument();
  });
});

describe("SourcesRow", () => {
  const mockSources: Source[] = [
    { id: "comp1", type: "company", label: "Acme Corp" },
    { id: "doc1", type: "doc", label: "User Guide" },
    { id: "act1", type: "activity", label: "Meeting" },
  ];

  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders collapsed by default", () => {
    render(<SourcesRow sources={mockSources} />);

    expect(screen.getByText("Sources (3)")).toBeInTheDocument();
    expect(screen.queryByText("Acme Corp")).not.toBeInTheDocument();
  });

  it("shows expand arrow when collapsed", () => {
    render(<SourcesRow sources={mockSources} />);

    expect(screen.getByText("▶")).toBeInTheDocument();
  });

  it("expands when header is clicked", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button", { name: /Sources \(3\)/ });
    fireEvent.click(header);

    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
    expect(screen.getByText("User Guide")).toBeInTheDocument();
    expect(screen.getByText("Meeting")).toBeInTheDocument();
  });

  it("shows collapse arrow when expanded", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    fireEvent.click(header);

    expect(screen.getByText("▼")).toBeInTheDocument();
  });

  it("collapses when header is clicked again", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    fireEvent.click(header);
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();

    fireEvent.click(header);
    expect(screen.queryByText("Acme Corp")).not.toBeInTheDocument();
  });

  it("renders all sources when expanded", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    fireEvent.click(header);

    const sourceChips = screen.getAllByRole("listitem");
    expect(sourceChips).toHaveLength(3);
  });

  // =========================================================================
  // Keyboard Interaction
  // =========================================================================

  it("expands on Enter key", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    fireEvent.keyDown(header, { key: "Enter" });

    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  it("expands on Space key", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    fireEvent.keyDown(header, { key: " " });

    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  it("does not expand on other keys", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    fireEvent.keyDown(header, { key: "a" });

    expect(screen.queryByText("Acme Corp")).not.toBeInTheDocument();
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("has proper ARIA attributes when collapsed", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    expect(header).toHaveAttribute("aria-expanded", "false");
    expect(header).toHaveAttribute("aria-controls", "sources-content");
  });

  it("has proper ARIA attributes when expanded", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    fireEvent.click(header);

    expect(header).toHaveAttribute("aria-expanded", "true");
  });

  it("has proper section label", () => {
    render(<SourcesRow sources={mockSources} />);

    expect(screen.getByLabelText("Sources referenced")).toBeInTheDocument();
  });

  it("sources content has proper role and label when expanded", () => {
    render(<SourcesRow sources={mockSources} />);

    const header = screen.getByRole("button");
    fireEvent.click(header);

    const content = screen.getByRole("list", { name: /3 sources referenced/ });
    expect(content).toHaveAttribute("id", "sources-content");
  });

  it("uses singular label for single source", () => {
    const singleSource: Source[] = [
      { id: "comp1", type: "company", label: "Acme Corp" },
    ];

    render(<SourcesRow sources={singleSource} />);

    const header = screen.getByRole("button");
    fireEvent.click(header);

    expect(screen.getByRole("list", { name: "1 source referenced" })).toBeInTheDocument();
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("returns null for empty sources array", () => {
    const { container } = render(<SourcesRow sources={[]} />);

    expect(container.firstChild).toBeNull();
  });

  it("handles single source", () => {
    const singleSource: Source[] = [
      { id: "comp1", type: "company", label: "Acme Corp" },
    ];

    render(<SourcesRow sources={singleSource} />);

    expect(screen.getByText("Sources (1)")).toBeInTheDocument();

    const header = screen.getByRole("button");
    fireEvent.click(header);

    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  it("handles many sources", () => {
    const manySources: Source[] = Array.from({ length: 50 }, (_, i) => ({
      id: `src${i}`,
      type: "company" as const,
      label: `Company ${i}`,
    }));

    render(<SourcesRow sources={manySources} />);

    expect(screen.getByText("Sources (50)")).toBeInTheDocument();

    const header = screen.getByRole("button");
    fireEvent.click(header);

    expect(screen.getByText("Company 0")).toBeInTheDocument();
    expect(screen.getByText("Company 49")).toBeInTheDocument();
  });

  it("handles duplicate source IDs", () => {
    const duplicateSources: Source[] = [
      { id: "comp1", type: "company", label: "First" },
      { id: "comp1", type: "company", label: "Second" },
    ];

    render(<SourcesRow sources={duplicateSources} />);

    const header = screen.getByRole("button");
    fireEvent.click(header);

    // Both should render despite duplicate IDs
    expect(screen.getByText("First")).toBeInTheDocument();
    expect(screen.getByText("Second")).toBeInTheDocument();
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("re-renders when sources change", () => {
    const sources1: Source[] = [
      { id: "comp1", type: "company", label: "Acme Corp" },
    ];

    const sources2: Source[] = [
      { id: "doc1", type: "doc", label: "User Guide" },
    ];

    const { rerender } = render(<SourcesRow sources={sources1} />);
    expect(screen.getByText("Sources (1)")).toBeInTheDocument();

    rerender(<SourcesRow sources={sources2} />);
    expect(screen.getByText("Sources (1)")).toBeInTheDocument();

    const header = screen.getByRole("button");
    fireEvent.click(header);

    expect(screen.queryByText("Acme Corp")).not.toBeInTheDocument();
    expect(screen.getByText("User Guide")).toBeInTheDocument();
  });
});
