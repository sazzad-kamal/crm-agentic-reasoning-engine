import React, { useState, useEffect, useMemo, useCallback, memo } from "react";
import { config } from "../config";

// =============================================================================
// Types
// =============================================================================

interface DataResponse {
  data: Record<string, unknown>[];
  total: number;
  columns: string[];
}

type DataTab = "companies" | "contacts" | "opportunities" | "activities" | "groups" | "history";

interface TabConfig {
  id: DataTab;
  label: string;
  icon: string;
  endpoint: string;
  nestedFields?: { key: string; label: string; icon: string }[];
}

const TABS: TabConfig[] = [
  { 
    id: "companies", 
    label: "Companies", 
    icon: "🏢", 
    endpoint: "/api/data/companies",
    nestedFields: [{ key: "_private_texts", label: "Notes & Attachments", icon: "📝" }]
  },
  { 
    id: "contacts", 
    label: "Contacts", 
    icon: "👤", 
    endpoint: "/api/data/contacts",
    nestedFields: [{ key: "_private_texts", label: "Notes & Attachments", icon: "📝" }]
  },
  { 
    id: "opportunities", 
    label: "Opportunities", 
    icon: "💰", 
    endpoint: "/api/data/opportunities",
    nestedFields: [
      { key: "_descriptions", label: "Description", icon: "📄" },
      { key: "_attachments", label: "Attachments", icon: "📎" }
    ]
  },
  { id: "activities", label: "Activities", icon: "📅", endpoint: "/api/data/activities" },
  { 
    id: "groups", 
    label: "Groups", 
    icon: "👥", 
    endpoint: "/api/data/groups",
    nestedFields: [{ key: "_members", label: "Members", icon: "🧑‍🤝‍🧑" }]
  },
  { id: "history", label: "History", icon: "📜", endpoint: "/api/data/history" },
];

// =============================================================================
// Hooks
// =============================================================================

function useDataFetch(endpoint: string) {
  const [data, setData] = useState<DataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(`${config.apiUrl}${endpoint}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [endpoint]);

  return { data, loading, error };
}

// =============================================================================
// Components
// =============================================================================

interface DataExplorerProps {
  onAskAbout?: (question: string) => void;
}

/**
 * Data Explorer - Browse all CRM data backing the AI
 */
export function DataExplorer({ onAskAbout }: DataExplorerProps) {
  const [activeTab, setActiveTab] = useState<DataTab>("companies");
  const [searchTerm, setSearchTerm] = useState("");

  const currentTab = TABS.find((t) => t.id === activeTab)!;
  const { data, loading, error } = useDataFetch(currentTab.endpoint);

  // Filter data based on search
  const filteredData = useMemo(() => {
    if (!data?.data || !searchTerm.trim()) return data?.data || [];

    const term = searchTerm.toLowerCase();
    return data.data.filter((row) =>
      Object.entries(row).some(([key, val]) => {
        // Don't search nested objects
        if (key.startsWith("_")) return false;
        return String(val).toLowerCase().includes(term);
      })
    );
  }, [data?.data, searchTerm]);

  const handleTabChange = useCallback((tabId: DataTab) => {
    setActiveTab(tabId);
    setSearchTerm("");
  }, []);

  const handleSearch = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  return (
    <div className="data-explorer">
      {/* Header */}
      <div className="data-explorer__header">
        <h2 className="data-explorer__title">
          <span className="data-explorer__icon">📊</span>
          Data Explorer
        </h2>
        <p className="data-explorer__subtitle">
          Browse the CRM data that powers the AI assistant
        </p>
      </div>

      {/* Tabs */}
      <div className="data-explorer__tabs" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`data-tab ${activeTab === tab.id ? "data-tab--active" : ""}`}
            onClick={() => handleTabChange(tab.id)}
          >
            <span className="data-tab__icon">{tab.icon}</span>
            <span className="data-tab__label">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Search */}
      <div className="data-explorer__search">
        <input
          type="text"
          className="data-explorer__search-input"
          placeholder={`Search ${currentTab.label.toLowerCase()}...`}
          value={searchTerm}
          onChange={handleSearch}
          aria-label={`Search ${currentTab.label}`}
        />
        {data && (
          <span className="data-explorer__count">
            {filteredData.length} of {data.total} records
          </span>
        )}
      </div>

      {/* Content */}
      <div className="data-explorer__content" role="tabpanel">
        {loading && <LoadingState />}
        {error && <ErrorState message={error} />}
        {!loading && !error && data && (
          <DataTable
            data={filteredData}
            columns={data.columns}
            onAskAbout={onAskAbout}
            dataType={activeTab}
            nestedFields={currentTab.nestedFields}
          />
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Sub-components
// =============================================================================

function LoadingState() {
  return (
    <div className="data-explorer__loading">
      <div className="data-explorer__spinner" />
      <span>Loading data...</span>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="data-explorer__error">
      <span className="data-explorer__error-icon">⚠️</span>
      <span>Failed to load data: {message}</span>
    </div>
  );
}

interface DataTableProps {
  data: Record<string, unknown>[];
  columns: string[];
  onAskAbout?: (question: string) => void;
  dataType: DataTab;
  nestedFields?: { key: string; label: string; icon: string }[];
}

const DataTable = memo(function DataTable({
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

  // Format column headers nicely
  const formatHeader = (col: string) =>
    col
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

  // Truncate long values
  const formatValue = (value: unknown) => {
    if (value === null || value === undefined) return "—";
    const str = String(value);
    if (str.length > 50) return str.slice(0, 50) + "...";
    return str;
  };

  // Generate a question about a row
  const getQuestion = (row: Record<string, unknown>) => {
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

  // Check if row has any nested data
  const hasNestedData = (row: Record<string, unknown>) => {
    if (!nestedFields) return false;
    return nestedFields.some((field) => {
      const nested = row[field.key] as unknown[] | undefined;
      return nested && nested.length > 0;
    });
  };

  // Filter out internal fields from columns display
  const displayColumns = columns.filter((col) => !col.startsWith("_"));

  return (
    <div className="data-table-container">
      <table className="data-table">
        <thead>
          <tr>
            {nestedFields && <th className="data-table__th data-table__th--expand"></th>}
            {onAskAbout && <th className="data-table__th data-table__th--action">Ask AI</th>}
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
            const hasNested = hasNestedData(row);
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
                        onClick={() => onAskAbout(getQuestion(row))}
                        title="Ask AI about this record"
                        aria-label="Ask AI about this record"
                      >
                        💬
                      </button>
                    </td>
                  )}
                  {displayColumns.map((col) => (
                    <td key={col} className="data-table__td" title={String(row[col] ?? "")}>
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

// =============================================================================
// Nested Data Display Component
// =============================================================================

interface NestedDataProps {
  row: Record<string, unknown>;
  nestedFields: { key: string; label: string; icon: string }[];
}

function NestedData({ row, nestedFields }: NestedDataProps) {
  return (
    <div className="nested-data">
      {nestedFields.map((field) => {
        const items = row[field.key] as Record<string, unknown>[] | undefined;
        if (!items || items.length === 0) return null;

        return (
          <div key={field.key} className="nested-data__section">
            <div className="nested-data__header">
              <span className="nested-data__icon">{field.icon}</span>
              <span className="nested-data__label">{field.label}</span>
              <span className="nested-data__count">({items.length})</span>
            </div>
            <div className="nested-data__items">
              {items.map((item, idx) => (
                <NestedItem key={idx} item={item} fieldKey={field.key} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

interface NestedItemProps {
  item: Record<string, unknown>;
  fieldKey: string;
}

function NestedItem({ item, fieldKey }: NestedItemProps) {
  // Different display based on type of nested data
  if (fieldKey === "_private_texts") {
    return (
      <div className="nested-item nested-item--text">
        <div className="nested-item__type">{String(item.metadata_type || item.type || "note")}</div>
        <div className="nested-item__content">{String(item.text || "")}</div>
        <div className="nested-item__meta">
          {item.metadata_file_name ? <span>📎 {String(item.metadata_file_name)}</span> : null}
          {item.metadata_created_at ? <span>🕐 {String(item.metadata_created_at)}</span> : null}
        </div>
      </div>
    );
  }

  if (fieldKey === "_descriptions") {
    return (
      <div className="nested-item nested-item--description">
        <div className="nested-item__title">{String(item.title || "Opportunity Notes")}</div>
        <div className="nested-item__content">{String(item.text || "")}</div>
        <div className="nested-item__meta">
          {item.created_at ? <span>Created: {String(item.created_at)}</span> : null}
        </div>
      </div>
    );
  }

  if (fieldKey === "_attachments") {
    return (
      <div className="nested-item nested-item--attachment">
        <span className="nested-item__icon">📎</span>
        <span className="nested-item__filename">{String(item.file_name || item.name || "Attachment")}</span>
        <span className="nested-item__size">{String(item.file_size || "")}</span>
      </div>
    );
  }

  if (fieldKey === "_members") {
    return (
      <div className="nested-item nested-item--member">
        <span className="nested-item__icon">🏢</span>
        <span className="nested-item__id">Company: {String(item.company_id || "")}</span>
        {item.contact_id ? <span className="nested-item__contact">Contact: {String(item.contact_id)}</span> : null}
        <span className="nested-item__date">Added: {String(item.added_at || "")}</span>
      </div>
    );
  }

  // Generic fallback
  return (
    <div className="nested-item">
      {Object.entries(item)
        .filter(([key]) => !key.startsWith("_"))
        .slice(0, 5)
        .map(([key, val]) => (
          <span key={key} className="nested-item__field">
            <strong>{key}:</strong> {String(val)}
          </span>
        ))}
    </div>
  );
}

export default DataExplorer;
