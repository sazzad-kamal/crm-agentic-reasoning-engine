import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { NestedData } from "../components/dataExplorer/NestedDataDisplay";

describe("NestedDataDisplay", () => {
  // =========================================================================
  // NestedData Component
  // =========================================================================

  describe("NestedData", () => {
    it("renders nothing when no nested fields have data", () => {
      const { container } = render(
        <NestedData row={{}} nestedFields={[]} />
      );
      expect(container.querySelector(".nested-data__section")).not.toBeInTheDocument();
    });

    it("renders nothing when nested field array is empty", () => {
      const row = { _private_texts: [] };
      const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

      const { container } = render(
        <NestedData row={row} nestedFields={nestedFields} />
      );
      expect(container.querySelector(".nested-data__section")).not.toBeInTheDocument();
    });

    it("renders section header with icon, label, and count", () => {
      const row = {
        _private_texts: [{ text: "Note 1" }, { text: "Note 2" }],
      };
      const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText("📝")).toBeInTheDocument();
      expect(screen.getByText("Notes")).toBeInTheDocument();
      expect(screen.getByText("(2)")).toBeInTheDocument();
    });
  });

  // =========================================================================
  // PrivateTextItem
  // =========================================================================

  describe("PrivateTextItem", () => {
    it("renders text content", () => {
      const row = {
        _private_texts: [{ text: "Important note content" }],
      };
      const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText("Important note content")).toBeInTheDocument();
    });

    it("renders metadata type", () => {
      const row = {
        _private_texts: [{ text: "Content", metadata_type: "call_note" }],
      };
      const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText("call_note")).toBeInTheDocument();
    });

    it("renders file name when present", () => {
      const row = {
        _private_texts: [{ text: "Content", metadata_file_name: "document.pdf" }],
      };
      const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText(/document.pdf/)).toBeInTheDocument();
    });

    it("renders created_at timestamp", () => {
      const row = {
        _private_texts: [{ text: "Content", metadata_created_at: "2025-01-01" }],
      };
      const nestedFields = [{ key: "_private_texts", label: "Notes", icon: "📝" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText(/2025-01-01/)).toBeInTheDocument();
    });
  });

  // =========================================================================
  // MemberItem
  // =========================================================================

  describe("MemberItem", () => {
    it("renders member with company_id", () => {
      const row = {
        _members: [{ company_id: "ACME-001", added_at: "2025-01-01" }],
      };
      const nestedFields = [{ key: "_members", label: "Members", icon: "👥" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText(/Company: ACME-001/)).toBeInTheDocument();
    });

    it("renders contact_id when present", () => {
      const row = {
        _members: [{ company_id: "ACME-001", contact_id: "CONTACT-123", added_at: "2025-01-01" }],
      };
      const nestedFields = [{ key: "_members", label: "Members", icon: "👥" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText(/Contact: CONTACT-123/)).toBeInTheDocument();
    });

    it("renders added_at date", () => {
      const row = {
        _members: [{ company_id: "ACME-001", added_at: "2025-02-15" }],
      };
      const nestedFields = [{ key: "_members", label: "Members", icon: "👥" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText(/Added: 2025-02-15/)).toBeInTheDocument();
    });
  });

  // =========================================================================
  // GenericItem (fallback for unknown field types)
  // =========================================================================

  describe("GenericItem", () => {
    it("renders unknown field type with GenericItem", () => {
      const row = {
        _custom_data: [{ field1: "value1", field2: "value2" }],
      };
      const nestedFields = [{ key: "_custom_data", label: "Custom", icon: "⭐" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText(/field1:/)).toBeInTheDocument();
      expect(screen.getByText(/value1/)).toBeInTheDocument();
      expect(screen.getByText(/field2:/)).toBeInTheDocument();
      expect(screen.getByText(/value2/)).toBeInTheDocument();
    });

    it("limits generic item to 5 fields", () => {
      const row = {
        _custom_data: [{
          field1: "v1",
          field2: "v2",
          field3: "v3",
          field4: "v4",
          field5: "v5",
          field6: "v6", // Should not appear
          field7: "v7", // Should not appear
        }],
      };
      const nestedFields = [{ key: "_custom_data", label: "Custom", icon: "⭐" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText(/field1:/)).toBeInTheDocument();
      expect(screen.getByText(/field5:/)).toBeInTheDocument();
      expect(screen.queryByText(/field6:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/field7:/)).not.toBeInTheDocument();
    });

    it("filters out fields starting with underscore", () => {
      const row = {
        _custom_data: [{ visible: "yes", _hidden: "no" }],
      };
      const nestedFields = [{ key: "_custom_data", label: "Custom", icon: "⭐" }];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText(/visible:/)).toBeInTheDocument();
      expect(screen.queryByText(/_hidden:/)).not.toBeInTheDocument();
    });
  });

  // =========================================================================
  // Multiple Nested Fields
  // =========================================================================

  describe("Multiple Nested Fields", () => {
    it("renders multiple nested field sections", () => {
      const row = {
        _private_texts: [{ text: "Note 1" }],
        _members: [{ company_id: "ACME-001", added_at: "2025-01-01" }],
      };
      const nestedFields = [
        { key: "_private_texts", label: "Notes", icon: "📝" },
        { key: "_members", label: "Members", icon: "👥" },
      ];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText("Notes")).toBeInTheDocument();
      expect(screen.getByText("Members")).toBeInTheDocument();
    });

    it("only renders sections with data", () => {
      const row = {
        _private_texts: [{ text: "Note 1" }],
        _members: [], // Empty, should not render
      };
      const nestedFields = [
        { key: "_private_texts", label: "Notes", icon: "📝" },
        { key: "_members", label: "Members", icon: "👥" },
      ];

      render(<NestedData row={row} nestedFields={nestedFields} />);

      expect(screen.getByText("Notes")).toBeInTheDocument();
      expect(screen.queryByText("Members")).not.toBeInTheDocument();
    });
  });
});
