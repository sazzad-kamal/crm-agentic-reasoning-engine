import { describe, it, expect } from "vitest";
import { formatDate, formatDateTime, formatCurrency } from "../utils/formatters";

describe("formatters", () => {
  // =========================================================================
  // formatDate
  // =========================================================================

  describe("formatDate", () => {
    it("formats valid ISO date string", () => {
      const result = formatDate("2024-01-15");
      expect(result).toMatch(/Jan/);
      expect(result).toMatch(/1[45]/); // Could be 14 or 15 depending on timezone
      expect(result).toMatch(/2024/);
    });

    it("formats date with time component", () => {
      const result = formatDate("2024-12-25T10:30:00");
      expect(result).toMatch(/Dec/);
      expect(result).toMatch(/25/);
      expect(result).toMatch(/2024/);
    });

    it("handles dates with timezone", () => {
      const result = formatDate("2024-06-01T00:00:00Z");
      expect(result).toMatch(/2024/);
    });

    it("returns original string for invalid date", () => {
      const invalid = "not-a-date";
      const result = formatDate(invalid);
      expect(result).toBe(invalid);
    });

    it("handles empty string", () => {
      const result = formatDate("");
      expect(result).toBe("");
    });

    it("handles malformed date string", () => {
      const result = formatDate("2024-99-99");
      expect(typeof result).toBe("string");
    });

    it("formats different months correctly", () => {
      const jan = formatDate("2024-01-01T12:00:00Z");
      const dec = formatDate("2024-12-31T12:00:00Z");

      expect(jan).toMatch(/Jan/);
      expect(dec).toMatch(/Dec/);
    });

    it("handles leap year dates", () => {
      const result = formatDate("2024-02-29T12:00:00Z");
      expect(result).toMatch(/Feb/);
      expect(result).toMatch(/2[89]/); // Could be 28 or 29 depending on timezone
    });
  });

  // =========================================================================
  // formatDateTime
  // =========================================================================

  describe("formatDateTime", () => {
    it("formats valid ISO datetime string", () => {
      const result = formatDateTime("2024-01-15T14:30:00");

      // Should include date components
      expect(result).toMatch(/Jan/);
      expect(result).toMatch(/15/);
      expect(result).toMatch(/2024/);

      // Should include time components
      expect(result).toMatch(/:/); // Time separator
    });

    it("formats datetime with timezone", () => {
      const result = formatDateTime("2024-06-01T12:00:00Z");
      expect(result).toMatch(/2024/);
      expect(result).toMatch(/:/);
    });

    it("includes hours and minutes", () => {
      const result = formatDateTime("2024-01-15T09:45:00");

      // Format varies by locale, but should have time
      expect(result).toMatch(/\d{1,2}:\d{2}/); // HH:MM pattern
    });

    it("handles midnight", () => {
      const result = formatDateTime("2024-01-15T00:00:00");
      expect(result).toMatch(/2024/);
    });

    it("handles noon", () => {
      const result = formatDateTime("2024-01-15T12:00:00");
      expect(result).toMatch(/2024/);
    });

    it("returns original string for invalid datetime", () => {
      const invalid = "invalid-datetime";
      const result = formatDateTime(invalid);
      expect(result).toBe(invalid);
    });

    it("handles empty string", () => {
      const result = formatDateTime("");
      expect(result).toBe("");
    });

    it("formats different times of day", () => {
      const morning = formatDateTime("2024-01-15T08:30:00");
      const evening = formatDateTime("2024-01-15T20:30:00");

      expect(morning).toMatch(/\d{1,2}:\d{2}/);
      expect(evening).toMatch(/\d{1,2}:\d{2}/);
    });
  });

  // =========================================================================
  // formatCurrency
  // =========================================================================

  describe("formatCurrency", () => {
    it("formats positive integer", () => {
      const result = formatCurrency(1000);
      expect(result).toMatch(/1,000/);
      expect(result).toMatch(/\$/);
    });

    it("formats large numbers with commas", () => {
      const result = formatCurrency(1000000);
      expect(result).toMatch(/1,000,000/);
    });

    it("formats zero", () => {
      const result = formatCurrency(0);
      expect(result).toMatch(/\$0/);
    });

    it("formats negative numbers", () => {
      const result = formatCurrency(-500);
      expect(result).toContain("-");
      expect(result).toMatch(/500/);
    });

    it("rounds to no decimal places", () => {
      const result = formatCurrency(1234.56);
      expect(result).toMatch(/1,235/); // Rounded up
      expect(result).not.toMatch(/\./); // No decimals
    });

    it("formats small numbers", () => {
      const result = formatCurrency(99);
      expect(result).toMatch(/99/);
    });

    it("formats very large numbers", () => {
      const result = formatCurrency(999999999);
      expect(result).toMatch(/999,999,999/);
    });

    it("includes dollar sign", () => {
      const result = formatCurrency(100);
      expect(result).toMatch(/\$/);
    });

    it("handles decimal rounding correctly", () => {
      expect(formatCurrency(1234.4)).toMatch(/1,234/); // Rounds down
      expect(formatCurrency(1234.5)).toMatch(/1,235/); // Rounds up
      expect(formatCurrency(1234.9)).toMatch(/1,235/); // Rounds up
    });

    it("formats round thousands", () => {
      expect(formatCurrency(1000)).toMatch(/1,000/);
      expect(formatCurrency(10000)).toMatch(/10,000/);
      expect(formatCurrency(100000)).toMatch(/100,000/);
    });
  });

  // =========================================================================
  // Integration & Edge Cases
  // =========================================================================

  describe("edge cases", () => {
    it("handles all formatters with typical CRM data", () => {
      // Typical opportunity data
      const date = formatDate("2024-12-31");
      const datetime = formatDateTime("2024-12-31T23:59:59");
      const value = formatCurrency(50000);

      expect(date).toBeTruthy();
      expect(datetime).toBeTruthy();
      expect(value).toBeTruthy();
    });

    it("handles formatters with null-like values gracefully", () => {
      // These should not throw
      expect(() => formatDate("")).not.toThrow();
      expect(() => formatDateTime("")).not.toThrow();
      expect(() => formatCurrency(0)).not.toThrow();
    });

    it("formatCurrency handles Infinity", () => {
      const result = formatCurrency(Infinity);
      expect(typeof result).toBe("string");
    });

    it("formatCurrency handles -Infinity", () => {
      const result = formatCurrency(-Infinity);
      expect(typeof result).toBe("string");
    });

    it("formatCurrency handles NaN", () => {
      const result = formatCurrency(NaN);
      expect(typeof result).toBe("string");
    });

    it("formatDate handles future dates", () => {
      const result = formatDate("2099-12-31");
      expect(result).toMatch(/2099/);
    });

    it("formatDate handles very old dates", () => {
      const result = formatDate("1900-01-01T12:00:00Z");
      expect(result).toMatch(/19(00|99)/); // Could be 1899 or 1900 depending on timezone
    });

    it("formatCurrency handles fractional cents", () => {
      const result = formatCurrency(99.999);
      expect(result).toMatch(/100/); // Rounds to 100
    });

    it("formats are consistent", () => {
      // Same input should produce same output
      const date1 = formatDate("2024-01-15");
      const date2 = formatDate("2024-01-15");
      expect(date1).toBe(date2);

      const curr1 = formatCurrency(1000);
      const curr2 = formatCurrency(1000);
      expect(curr1).toBe(curr2);
    });
  });

  // =========================================================================
  // Locale Independence
  // =========================================================================

  describe("locale handling", () => {
    it("formatCurrency uses USD", () => {
      const result = formatCurrency(100);
      // Should contain $ (USD symbol)
      expect(result).toMatch(/\$/);
    });

    it("date formats use locale defaults", () => {
      // Date format will vary by locale, but should always include components
      const result = formatDate("2024-06-15T12:00:00Z");

      expect(result).toMatch(/2024/); // Year
      expect(result).toMatch(/1[45]/);   // Day (could be 14 or 15 depending on timezone)
      // Month format varies (Jun, June, 6, etc.) so we don't test specific format
    });

    it("handles dates consistently across tests", () => {
      // Multiple calls should produce consistent results
      const results = [
        formatDate("2024-01-01"),
        formatDate("2024-01-01"),
        formatDate("2024-01-01"),
      ];

      expect(results[0]).toBe(results[1]);
      expect(results[1]).toBe(results[2]);
    });
  });
});
