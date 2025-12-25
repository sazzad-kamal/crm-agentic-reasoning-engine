/**
 * Data table component for displaying CRM records with expandable nested data.
 */
import React, { useState, memo } from "react";
import type { DataTab, NestedFieldConfig } from "./types";
import { NestedData } from "./NestedDataDisplay";

// =============================================================================
// Types
// =============================================================================

interface DataTableProps {
  data: Record<string, unknown>[];
  columns: string[];
  onAskAbout?: (question: string) => void;
  dataType: DataTab;
  nestedFields?: NestedFieldConfig[];
}

// =============================================================================
// Utilities
// =============================================================================

/** Format column headers nicely (snake_case -> Title Case) */
const formatHeader = (col: string): string =>
  col.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/** Truncate long values for display */
const formatValue = (value: unknown): string => {
  if (value === null || value === undefined) return "—";
  const str = String(value);
  return str.length > 50 ? str.slice(0, 50) + "..." : str;
};

/** Generate AI question based on data type and row content */
const getQuestion = (
  dataType: DataTab,
  row: Record<string, unknown>,
  columns: string[]
): string => {
  switch (dataType) {
    case "companies":
      return `What's happening with ${row.name || row.company_id}?`;
    case "contacts":
      return `Tell me about ${row.first_name} ${row.last_name} at ${row.company_id}`;
    case "opportunities":
      return `What's the status of the ${row.name} opportunity?`;
    case "activities":
      return `What activities are scheduled for ${row.company_id}?`;
    case "groups":
      return `Tell me about the ${row.name} group`;
    case "history":
      return `Show me the history for ${row.company_id}`;
    default:
      return `Tell me about ${row[columns[0]]}`;
  }
};

/** Check if row has any nested data */
const hasNestedData = (
  row: Record<string, unknown>,
  nestedFields?: NestedFieldConfig[]
): boolean => {
  if (!nestedFields) return false;
  return nestedFields.some((field) => {
    const nested = row[field.key] as unknown[] | undefined;
    return nested && nested.length > 0;
  });
};

// =============================================================================
// Component
// =============================================================================

export const DataTable = memo(function DataTable({
  data,
  columns,
  onAskAbout,
  dataType,
  nestedFields,
}: DataTableProps) {
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());

  if (data.length === 0) {
    return (
      <div className="data-table__empty">
        <span>No records found</span>
      </div>
    );
  }

  const toggleRow = (idx: number) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  // Filter out internal fields from columns display
  const displayColumns = columns.filter((col) => !col.startsWith("_"));

  return (
    <div className="data-table-container">
      <table className="data-table">
        <thead>
          <tr>
            {nestedFields && (
              <th className="data-table__th data-table__th--expand"></th>
            )}
            {onAskAbout && (
              <th className="data-table__th data-table__th--action">Ask AI</th>
            )}
            {displayColumns.map((col) => (
              <th key={col} className="data-table__th">
                {formatHeader(col)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => {
            const isExpanded = expandedRows.has(idx);
            const hasNested = hasNestedData(row, nestedFields);
            return (
              <React.Fragment key={idx}>
                <tr
                  className={`data-table__row ${hasNested ? "data-table__row--expandable" : ""} ${isExpanded ? "data-table__row--expanded" : ""}`}
                >
                  {nestedFields && (
                    <td className="data-table__td data-table__td--expand">
                      {hasNested && (
                        <button
                          className="data-table__expand-btn"
                          onClick={() => toggleRow(idx)}
                          aria-expanded={isExpanded}
                          aria-label={isExpanded ? "Collapse" : "Expand"}
                        >
                          {isExpanded ? "▼" : "▶"}
                        </button>
                      )}
                    </td>
                  )}
                  {onAskAbout && (
                    <td className="data-table__td data-table__td--action">
                      <button
                        className="data-table__ask-btn"
                        onClick={() =>
                          onAskAbout(getQuestion(dataType, row, columns))
                        }
                        title="Ask AI about this record"
                        aria-label="Ask AI about this record"
                      >
                        💬
                      </button>
                    </td>
                  )}
                  {displayColumns.map((col) => (
                    <td
                      key={col}
                      className="data-table__td"
                      title={String(row[col] ?? "")}
                    >
                      {formatValue(row[col])}
                    </td>
                  ))}
                </tr>
                {isExpanded && hasNested && nestedFields && (
                  <tr className="data-table__nested-row">
                    <td colSpan={displayColumns.length + (onAskAbout ? 2 : 1)}>
                      <NestedData row={row} nestedFields={nestedFields} />
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
});
