import { describe, it, expect } from "vitest";

describe("barrel exports", () => {
  describe("components/index.ts", () => {
    it("exports all public components", async () => {
      const components = await import("../components");

      expect(components.LoadingDots).toBeDefined();
      expect(components.FollowUpSuggestions).toBeDefined();
      expect(components.DataTables).toBeDefined();
      expect(components.MessageBlock).toBeDefined();
      expect(components.ChatArea).toBeDefined();
      expect(components.InputBar).toBeDefined();
      expect(components.ErrorBanner).toBeDefined();
      expect(components.ErrorBoundary).toBeDefined();
      expect(components.SkipLink).toBeDefined();
      expect(components.Avatar).toBeDefined();
      expect(components.CopyButton).toBeDefined();
      expect(components.MarkdownText).toBeDefined();
      expect(components.DataExplorer).toBeDefined();
    });
  });

  describe("components/dataExplorer/index.ts", () => {
    it("exports DataExplorer and DataTable", async () => {
      const dataExplorer = await import("../components/dataExplorer");

      expect(dataExplorer.DataExplorer).toBeDefined();
      expect(dataExplorer.DataTable).toBeDefined();
      expect(dataExplorer.NestedData).toBeDefined();
    });

    it("exports types and config", async () => {
      const dataExplorer = await import("../components/dataExplorer");

      expect(dataExplorer.TABS).toBeDefined();
      expect(Array.isArray(dataExplorer.TABS)).toBe(true);
    });
  });

  describe("components/dataExplorer/types.ts (re-export)", () => {
    it("re-exports types from centralized location", async () => {
      const types = await import("../components/dataExplorer/types");

      expect(types.TABS).toBeDefined();
      expect(Array.isArray(types.TABS)).toBe(true);
      expect(types.TABS.length).toBe(5);
    });
  });
});
