/**
 * Data table component for displaying CRM records with expandable nested data.
 * Features: pagination, sorting indication, expandable rows, accessibility.
 */
import React, { useState, useMemo, useCallback, memo, useId } from "react";
import type { DataTab, NestedFieldConfig } from "../../types/dataExplorer";
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
  /** Number of rows per page (default: 10) */
  pageSize?: number;
}

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  tableId: string;
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
// Pagination Component
// =============================================================================

const Pagination = memo(function Pagination({
  currentPage,
  totalPages,
  totalItems,
  pageSize,
  onPageChange,
  tableId,
}: PaginationProps) {
  const startItem = (currentPage - 1) * pageSize + 1;
  const endItem = Math.min(currentPage * pageSize, totalItems);

  // Generate page numbers to display
  const getPageNumbers = (): (number | "...")[] => {
    const pages: (number | "...")[] = [];
    const maxVisible = 5;

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Always show first page
      pages.push(1);

      if (currentPage > 3) {
        pages.push("...");
      }

      // Show pages around current
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);

      for (let i = start; i <= end; i++) {
        pages.push(i);
      }

      if (currentPage < totalPages - 2) {
        pages.push("...");
      }

      // Always show last page
      pages.push(totalPages);
    }

    return pages;
  };

  if (totalPages <= 1) return null;

  return (
    <nav
      className="pagination"
      aria-label={`Pagination for ${tableId}`}
      role="navigation"
    >
      <div className="pagination__info" aria-live="polite">
        Showing {startItem}–{endItem} of {totalItems} items
      </div>
      <div className="pagination__controls">
        <button
          className="pagination__btn pagination__btn--prev"
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          aria-label="Go to previous page"
        >
          ← Prev
        </button>

        <div className="pagination__pages" role="list">
          {getPageNumbers().map((page, idx) =>
            page === "..." ? (
              <span key={`ellipsis-${idx}`} className="pagination__ellipsis" aria-hidden="true">
                ...
              </span>
            ) : (
              <button
                key={page}
                className={`pagination__page ${currentPage === page ? "pagination__page--active" : ""}`}
                onClick={() => onPageChange(page)}
                aria-label={`Go to page ${page}`}
                aria-current={currentPage === page ? "page" : undefined}
              >
                {page}
              </button>
            )
          )}
        </div>

        <button
          className="pagination__btn pagination__btn--next"
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          aria-label="Go to next page"
        >
          Next →
        </button>
      </div>
    </nav>
  );
});

// =============================================================================
// Main Component
// =============================================================================

export const DataTable = memo(function DataTable({
  data,
  columns,
  onAskAbout,
  dataType,
  nestedFields,
  pageSize = 10,
}: DataTableProps) {
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  const [requestedPage, setRequestedPage] = useState(1);
  const tableId = useId();

  // Calculate pagination
  const totalItems = data.length;
  const totalPages = Math.ceil(totalItems / pageSize);

  // Derive the actual current page (clamp to valid range)
  const currentPage = useMemo(() => {
    if (totalPages === 0) return 1;
    return Math.min(Math.max(1, requestedPage), totalPages);
  }, [requestedPage, totalPages]);

  // Get current page data
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return data.slice(startIndex, endIndex);
  }, [data, currentPage, pageSize]);

  // Handle page change request
  const handlePageChange = useCallback((page: number) => {
    setRequestedPage(page);
    setExpandedRows(new Set()); // Collapse all rows when changing pages
  }, []);

  if (data.length === 0) {
    return (
      <div className="data-table__empty" role="status" aria-live="polite">
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
  const colCount = displayColumns.length + (onAskAbout ? 1 : 0) + (nestedFields ? 1 : 0);

  return (
    <div className="data-table-wrapper">
      <div
        className="data-table-container"
        role="region"
        aria-label={`${dataType} data table`}
        tabIndex={0}
      >
        <table
          className="data-table"
          aria-describedby={`${tableId}-caption`}
        >
          <caption id={`${tableId}-caption`} className="visually-hidden">
            {dataType.charAt(0).toUpperCase() + dataType.slice(1)} data with {totalItems} records
          </caption>
          <thead>
            <tr>
              {nestedFields && (
                <th
                  className="data-table__th data-table__th--expand"
                  scope="col"
                  aria-label="Expand row"
                />
              )}
              {onAskAbout && (
                <th
                  className="data-table__th data-table__th--action"
                  scope="col"
                >
                  Ask AI
                </th>
              )}
              {displayColumns.map((col) => (
                <th key={col} className="data-table__th" scope="col">
                  {formatHeader(col)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, idx) => {
              const globalIdx = (currentPage - 1) * pageSize + idx;
              const isExpanded = expandedRows.has(idx);
              const hasNested = hasNestedData(row, nestedFields);
              const rowId = `${tableId}-row-${idx}`;

              return (
                <React.Fragment key={globalIdx}>
                  <tr
                    id={rowId}
                    className={`data-table__row ${hasNested ? "data-table__row--expandable" : ""} ${isExpanded ? "data-table__row--expanded" : ""}`}
                    aria-expanded={hasNested ? isExpanded : undefined}
                  >
                    {nestedFields && (
                      <td className="data-table__td data-table__td--expand">
                        {hasNested && (
                          <button
                            className="data-table__expand-btn"
                            onClick={() => toggleRow(idx)}
                            aria-expanded={isExpanded}
                            aria-controls={`${rowId}-nested`}
                            aria-label={isExpanded ? "Collapse details" : "Expand details"}
                          >
                            <span aria-hidden="true">
                              {isExpanded ? "▼" : "▶"}
                            </span>
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
                          aria-label={`Ask AI about ${row.name || row.company_id || "this record"}`}
                        >
                          <span aria-hidden="true">💬</span>
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
                    <tr
                      id={`${rowId}-nested`}
                      className="data-table__nested-row"
                      aria-labelledby={rowId}
                    >
                      <td colSpan={colCount}>
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

      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        totalItems={totalItems}
        pageSize={pageSize}
        onPageChange={handlePageChange}
        tableId={dataType}
      />
    </div>
  );
});
