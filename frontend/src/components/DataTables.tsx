import { useState, useCallback, useMemo, memo } from "react";
import type { KeyboardEvent } from "react";
import type { RawData } from "../types";
import { formatDate, formatDateTime, formatCurrency } from "../utils/formatters";

/** Keyboard handler for Enter/Space activation (accessibility) */
function createActivationHandler<T extends HTMLElement>(
  action: () => void
): (e: KeyboardEvent<T>) => void {
  return (e: KeyboardEvent<T>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      action();
    }
  };
}

// Keys handled by legacy tables or internal (skip in dynamic rendering)
const LEGACY_KEYS = new Set([
  "companies", "activities", "opportunities", "history", "renewals",
  "pipeline_summary", "data"
]);
const SKIP_KEYS = new Set(["duckdb", "error"]);

// Friendly labels for known demo mode keys
const KEY_LABELS: Record<string, string> = {
  today_meetings: "Today's Meetings",
  recent_history: "Recent History",
  open_opportunities: "Open Opportunities",
  overdue_followups: "Overdue Follow-ups",
  forecast_30d: "30-Day Forecast",
  forecast_60d: "60-Day Forecast",
  forecast_90d: "90-Day Forecast",
  slipped_deals: "Slipped Deals",
  at_risk_deals: "At-Risk Deals",
  expand: "Expand (High Engagement)",
  save: "Save (At Risk)",
  reactivate: "Re-activate",
  relationship_analysis: "Relationship Analysis",
  total_pipeline: "Total Pipeline",
  total_weighted: "Weighted Pipeline",
  at_risk_pct: "At-Risk %",
  no_date_count: "Missing Close Date",
};

// Icons for entity groups
const KEY_ICONS: Record<string, string> = {
  today_meetings: "📅",
  recent_history: "📜",
  open_opportunities: "💰",
  overdue_followups: "⚠️",
  forecast_30d: "📊",
  forecast_60d: "📊",
  forecast_90d: "📊",
  slipped_deals: "⏰",
  at_risk_deals: "🚨",
  expand: "📈",
  save: "🛡️",
  reactivate: "🔄",
  relationship_analysis: "🤝",
};

/** Format snake_case key to Title Case */
function formatKey(key: string): string {
  if (KEY_LABELS[key]) return KEY_LABELS[key];
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Get icon for key */
function getIcon(key: string): string {
  return KEY_ICONS[key] || "📄";
}

/** Check if value is an array */
function isArray(val: unknown): val is unknown[] {
  return Array.isArray(val);
}

/** Check if value is an object (not array, not null) */
function isObject(val: unknown): val is Record<string, unknown> {
  return val !== null && typeof val === "object" && !Array.isArray(val);
}

/** Check if key should be hidden in table columns */
function isHiddenColumn(key: string): boolean {
  if (key.startsWith("_")) return true;
  if (key.endsWith("Id") || key.endsWith("ID")) return true;
  if (key === "id" || key === "duckdb") return true;
  return false;
}

/** Format cell value for display */
function formatCellValue(val: unknown, key: string): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") {
    // Format currency-like fields
    if (key.includes("value") || key.includes("pipeline") || key.includes("weighted") || key === "productTotal") {
      return formatCurrency(val);
    }
    // Format percentage
    if (key.includes("pct") || key.includes("probability")) {
      return `${val}%`;
    }
    return String(val);
  }
  if (typeof val === "boolean") return val ? "Yes" : "No";
  if (typeof val === "string") {
    // Try to detect and format dates
    if (/^\d{4}-\d{2}-\d{2}/.test(val)) {
      return formatDate(val);
    }
    return val || "—";
  }
  if (isArray(val)) return `${val.length} items`;
  if (isObject(val)) return "[Object]";
  return String(val);
}

/** Render a generic table from array of objects */
function GenericTable({ data, labelId }: { data: Record<string, unknown>[]; labelId: string }) {
  if (!data.length) return null;

  // Get visible columns from first row
  const columns = Object.keys(data[0]).filter(k => !isHiddenColumn(k));
  if (!columns.length) return null;

  return (
    <table className="data-table" aria-labelledby={labelId}>
      <thead>
        <tr>
          {columns.map((col) => (
            <th key={col} scope="col">{formatKey(col)}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, idx) => (
          <tr key={idx}>
            {columns.map((col) => {
              const val = formatCellValue(row[col], col);
              // Truncate long text
              if (val.length > 80) {
                return <td key={col} title={val}>{val.slice(0, 80)}…</td>;
              }
              return <td key={col}>{val}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Render scalar metrics as key-value pairs */
function ScalarMetrics({ metrics }: { metrics: [string, unknown][] }) {
  return (
    <table className="data-table data-table--metrics">
      <tbody>
        {metrics.map(([key, val]) => (
          <tr key={key}>
            <th scope="row">{formatKey(key)}</th>
            <td>{formatCellValue(val, key)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

interface DataTablesProps {
  rawData: RawData;
}

/**
 * Collapsible data tables showing raw CRM data.
 * Supports both legacy SQL mode entities and dynamic demo mode entity groups.
 * Memoized for performance with large datasets.
 */
export const DataTables = memo(function DataTables({ rawData }: DataTablesProps) {
  const [expanded, setExpanded] = useState(false);

  // Separate dynamic keys from legacy keys
  const { dynamicEntries, scalarMetrics } = useMemo(() => {
    const entries: [string, unknown][] = [];
    const scalars: [string, unknown][] = [];

    for (const [key, val] of Object.entries(rawData)) {
      if (LEGACY_KEYS.has(key) || SKIP_KEYS.has(key) || key.startsWith("_")) continue;

      if (isArray(val) && val.length > 0) {
        entries.push([key, val]);
      } else if (isObject(val)) {
        // Object with nested data (like forecast_30d with deals[])
        const nested = val as Record<string, unknown>;
        // Check for deals array inside
        if (isArray(nested.deals) && nested.deals.length > 0) {
          entries.push([key, nested.deals]);
        }
        // Collect scalar summaries from the object
        for (const [k, v] of Object.entries(nested)) {
          if (k !== "deals" && typeof v === "number") {
            scalars.push([`${key}_${k}`, v]);
          }
        }
      } else if (typeof val === "number" || typeof val === "string") {
        // Scalar value (total_pipeline, at_risk_pct, etc.)
        scalars.push([key, val]);
      }
    }

    return { dynamicEntries: entries, scalarMetrics: scalars };
  }, [rawData]);

  const hasLegacyData = useMemo(
    () =>
      (rawData.companies && rawData.companies.length > 0) ||
      (rawData.activities && rawData.activities.length > 0) ||
      (rawData.opportunities && rawData.opportunities.length > 0) ||
      (rawData.history && rawData.history.length > 0) ||
      (rawData.renewals && rawData.renewals.length > 0) ||
      (rawData.data && rawData.data.length > 0) ||
      rawData.pipeline_summary,
    [rawData]
  );

  const hasDynamicData = dynamicEntries.length > 0 || scalarMetrics.length > 0;
  const hasData = hasLegacyData || hasDynamicData;

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  const handleKeyDown = useMemo(
    () => createActivationHandler(toggleExpanded),
    [toggleExpanded]
  );

  // Build list of present data types for preview badges
  const presentDataTypes = useMemo(() => {
    const types: { key: string; label: string; count: number }[] = [];

    // Legacy types
    if (rawData.companies?.length) types.push({ key: "companies", label: "companies", count: rawData.companies.length });
    if (rawData.activities?.length) types.push({ key: "activities", label: "activities", count: rawData.activities.length });
    if (rawData.opportunities?.length) types.push({ key: "opportunities", label: "opportunities", count: rawData.opportunities.length });
    if (rawData.history?.length) types.push({ key: "history", label: "history", count: rawData.history.length });
    if (rawData.renewals?.length) types.push({ key: "renewals", label: "renewals", count: rawData.renewals.length });
    if (rawData.pipeline_summary) types.push({ key: "pipeline", label: "pipeline", count: 1 });
    if (rawData.data?.length) types.push({ key: "data", label: "records", count: rawData.data.length });

    // Dynamic types
    for (const [key, val] of dynamicEntries) {
      const arr = val as unknown[];
      types.push({ key, label: formatKey(key), count: arr.length });
    }

    if (scalarMetrics.length > 0) {
      types.push({ key: "metrics", label: "metrics", count: scalarMetrics.length });
    }

    return types;
  }, [rawData, dynamicEntries, scalarMetrics]);

  if (!hasData) return null;

  return (
    <section
      className="data-section"
      aria-label="Data used in response"
    >
      <button
        className="data-section__header"
        onClick={toggleExpanded}
        onKeyDown={handleKeyDown}
        type="button"
        aria-expanded={expanded}
        aria-controls="data-tables-content"
      >
        <span className="data-section__arrow" aria-hidden="true">{expanded ? "▼" : "▶"}</span>
        <span className="data-section__label">Data used</span>
        <span className="data-section__preview" aria-label={`Contains ${presentDataTypes.map(t => `${t.count} ${t.label}`).join(", ")}`}>
          {presentDataTypes.slice(0, 4).map((type) => (
            <span key={type.key} className="data-section__preview-badge" title={`${type.label} (${type.count})`}>
              {type.count} {type.label}
            </span>
          ))}
          {presentDataTypes.length > 4 && (
            <span className="data-section__preview-badge">+{presentDataTypes.length - 4} more</span>
          )}
        </span>
      </button>

      {expanded && (
        <div
          id="data-tables-content"
          className="data-section__content"
          role="region"
          aria-label="Data tables"
        >
          {/* ============================================================= */}
          {/* Dynamic Entity Groups (Demo Mode) */}
          {/* ============================================================= */}

          {/* Scalar Metrics Summary */}
          {scalarMetrics.length > 0 && (
            <div role="region" aria-label="Summary metrics" data-type="metrics">
              <h4 className="data-table__title" id="metrics-table-label">
                <span className="data-table__icon">📊</span>
                Summary Metrics
              </h4>
              <ScalarMetrics metrics={scalarMetrics} />
            </div>
          )}

          {/* Dynamic Array Tables */}
          {dynamicEntries.map(([key, val]) => {
            const arr = val as Record<string, unknown>[];
            const labelId = `dynamic-${key}-table-label`;
            return (
              <div key={key} role="region" aria-label={`${formatKey(key)} data`} data-type={key}>
                <h4 className="data-table__title" id={labelId}>
                  <span className="data-table__icon">{getIcon(key)}</span>
                  {formatKey(key)} ({arr.length})
                </h4>
                <GenericTable data={arr} labelId={labelId} />
              </div>
            );
          })}

          {/* ============================================================= */}
          {/* Legacy Tables (SQL Mode) */}
          {/* ============================================================= */}

          {/* Companies Table */}
          {rawData.companies && rawData.companies.length > 0 && (
            <div role="region" aria-label="Companies data" data-type="companies">
              <h4 className="data-table__title" id="companies-table-label">
                <span className="data-table__icon">🏢</span>
                Companies ({rawData.companies.length})
              </h4>
              <table className="data-table" aria-labelledby="companies-table-label">
                <thead>
                  <tr>
                    <th scope="col">Name</th>
                    <th scope="col">Plan</th>
                    <th scope="col">Renewal Date</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.companies.map((c) => (
                    <tr key={c.company_id}>
                      <td>{c.name}</td>
                      <td>{c.plan}</td>
                      <td>{formatDate(c.renewal_date)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Activities Table */}
          {rawData.activities && rawData.activities.length > 0 && (
            <div role="region" aria-label="Activities data" data-type="activities">
              <h4 className="data-table__title" id="activities-table-label">
                <span className="data-table__icon">📋</span>
                Activities ({rawData.activities.length})
              </h4>
              <table className="data-table" aria-labelledby="activities-table-label">
                <thead>
                  <tr>
                    <th scope="col">Type</th>
                    <th scope="col">Occurred At</th>
                    <th scope="col">Owner</th>
                    <th scope="col">Summary</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.activities.map((a) => (
                    <tr key={a.activity_id}>
                      <td>{a.type}</td>
                      <td>{formatDateTime(a.occurred_at)}</td>
                      <td>{a.owner}</td>
                      <td>{a.summary || a.subject || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Opportunities Table */}
          {rawData.opportunities && rawData.opportunities.length > 0 && (
            <div role="region" aria-label="Opportunities data" data-type="opportunities">
              <h4 className="data-table__title" id="opportunities-table-label">
                <span className="data-table__icon">💰</span>
                Opportunities ({rawData.opportunities.length})
              </h4>
              <table className="data-table" aria-labelledby="opportunities-table-label">
                <thead>
                  <tr>
                    <th scope="col">Name</th>
                    <th scope="col">Stage</th>
                    <th scope="col">Expected Close</th>
                    <th scope="col">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.opportunities.map((o) => (
                    <tr key={o.opportunity_id}>
                      <td>{o.name}</td>
                      <td>{o.stage}</td>
                      <td>{formatDate(o.expected_close_date)}</td>
                      <td>{formatCurrency(o.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* History Table */}
          {rawData.history && rawData.history.length > 0 && (
            <div role="region" aria-label="History data" data-type="history">
              <h4 className="data-table__title" id="history-table-label">
                <span className="data-table__icon">📜</span>
                History ({rawData.history.length})
              </h4>
              <table className="data-table" aria-labelledby="history-table-label">
                <thead>
                  <tr>
                    <th scope="col">Event</th>
                    <th scope="col">Occurred At</th>
                    <th scope="col">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.history.map((h) => (
                    <tr key={h.history_id}>
                      <td>{h.event_type}</td>
                      <td>{formatDateTime(h.occurred_at)}</td>
                      <td>{h.notes || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Renewals Table */}
          {rawData.renewals && rawData.renewals.length > 0 && (
            <div role="region" aria-label="Renewals data" data-type="renewals">
              <h4 className="data-table__title" id="renewals-table-label">
                <span className="data-table__icon">🔄</span>
                Upcoming Renewals ({rawData.renewals.length})
              </h4>
              <table className="data-table" aria-labelledby="renewals-table-label">
                <thead>
                  <tr>
                    <th scope="col">Company</th>
                    <th scope="col">Plan</th>
                    <th scope="col">Renewal Date</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.renewals.map((r) => (
                    <tr key={r.company_id}>
                      <td>{r.company_name}</td>
                      <td>{r.plan}</td>
                      <td>{formatDate(r.renewal_date)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pipeline Summary */}
          {rawData.pipeline_summary && (
            <div role="region" aria-label="Pipeline summary data" data-type="pipeline">
              <h4 className="data-table__title" id="pipeline-table-label">
                <span className="data-table__icon">📊</span>
                Pipeline Summary
              </h4>
              <table className="data-table" aria-labelledby="pipeline-table-label">
                <tbody>
                  <tr>
                    <th scope="row">Total Value</th>
                    <td>{formatCurrency(rawData.pipeline_summary.total_value)}</td>
                  </tr>
                  <tr>
                    <th scope="row">Opportunity Count</th>
                    <td>{rawData.pipeline_summary.count}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {/* Generic Data Table (raw query results) */}
          {rawData.data && rawData.data.length > 0 && (() => {
            const HIDDEN_COLS = new Set(["notes", "closed_date", "created_at", "updated_at", "currency", "probability", "source"]);
            const isVisible = (k: string) =>
              !k.startsWith("_") && !k.endsWith("_id") && !HIDDEN_COLS.has(k);

            // Check if data is from UNION ALL cross-table queries (source values are known types)
            const KNOWN_SOURCES = new Set(["company", "opportunity", "activity", "history", "contact"]);
            const firstSource = String(rawData.data[0].source ?? "");
            const isGroupedSource = "source" in rawData.data[0] && KNOWN_SOURCES.has(firstSource);

            if (isGroupedSource) {
              // Group rows by source for separate tables with proper column labels
              const SOURCE_CONFIG: Record<string, { title: string; icon: string; columns: { key: string; label: string }[] }> = {
                company:     { title: "Company",      icon: "🏢", columns: [{ key: "name", label: "Name" }, { key: "plan", label: "Plan" }, { key: "status", label: "Status" }, { key: "health_status", label: "Health" }, { key: "key_date", label: "Renewal Date" }, { key: "notes", label: "Notes" }] },
                opportunity: { title: "Opportunities", icon: "💰", columns: [{ key: "name", label: "Name" }, { key: "plan", label: "Stage" }, { key: "status", label: "Type" }, { key: "health_status", label: "Value" }, { key: "key_date", label: "Close Date" }, { key: "notes", label: "Notes" }] },
                activity:    { title: "Activities",    icon: "📋", columns: [{ key: "name", label: "Type" }, { key: "plan", label: "Subject" }, { key: "status", label: "Status" }, { key: "health_status", label: "Priority" }, { key: "key_date", label: "Due Date" }, { key: "notes", label: "Notes" }] },
                history:     { title: "History",       icon: "📜", columns: [{ key: "name", label: "Type" }, { key: "plan", label: "Subject" }, { key: "key_date", label: "Date" }, { key: "notes", label: "Notes" }] },
                contact:     { title: "Contacts",     icon: "👤", columns: [{ key: "name", label: "Name" }, { key: "plan", label: "Role" }, { key: "status", label: "Title" }, { key: "health_status", label: "Email" }, { key: "notes", label: "Notes" }] },
              };

              const grouped = new Map<string, Record<string, unknown>[]>();
              for (const row of rawData.data) {
                const src = String(row.source ?? "other");
                if (!grouped.has(src)) grouped.set(src, []);
                grouped.get(src)!.push(row);
              }

              return (
                <>
                  {[...grouped.entries()].map(([src, rows]) => {
                    const config = SOURCE_CONFIG[src];
                    if (!config) return null;
                    const columns = config.columns;
                    const labelId = `grouped-${src}-table-label`;
                    return (
                      <div key={src} role="region" aria-label={`${config.title} data`} data-type={src}>
                        <h4 className="data-table__title" id={labelId}>
                          <span className="data-table__icon">{config.icon}</span>
                          {config.title} ({rows.length})
                        </h4>
                        <table className="data-table" aria-labelledby={labelId}>
                          <thead>
                            <tr>
                              {columns.map((col) => (
                                <th key={col.key} scope="col">{col.label}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map((row, idx) => (
                              <tr key={idx}>
                                {columns.map((col) => {
                                  const raw = String(row[col.key] ?? "—");
                                  if (col.key === "notes" && raw.length > 80) {
                                    return <td key={col.key} title={raw}>{raw.slice(0, 80)}…</td>;
                                  }
                                  return <td key={col.key}>{raw}</td>;
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    );
                  })}
                </>
              );
            }

            return (
            <div role="region" aria-label="Query results" data-type="data">
              <h4 className="data-table__title" id="generic-table-label">
                <span className="data-table__icon">📄</span>
                Results ({rawData.data.length})
              </h4>
              <table className="data-table" aria-labelledby="generic-table-label">
                <thead>
                  <tr>
                    {Object.keys(rawData.data[0]).filter(isVisible).map((col) => (
                      <th key={col} scope="col">{col.replace(/_/g, " ")}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rawData.data.map((row, idx) => (
                    <tr key={idx}>
                      {Object.entries(row).filter(([k]) => isVisible(k)).map(([key, val]) => (
                        <td key={key}>{String(val ?? "—")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            );
          })()}
        </div>
      )}
    </section>
  );
});
