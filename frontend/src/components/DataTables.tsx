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

interface DataTablesProps {
  rawData: RawData;
}

/**
 * Collapsible data tables showing raw CRM data.
 * Memoized for performance with large datasets.
 */
export const DataTables = memo(function DataTables({ rawData }: DataTablesProps) {
  const [expanded, setExpanded] = useState(false);

  const hasData = useMemo(
    () =>
      (rawData.companies && rawData.companies.length > 0) ||
      (rawData.activities && rawData.activities.length > 0) ||
      (rawData.opportunities && rawData.opportunities.length > 0) ||
      (rawData.history && rawData.history.length > 0) ||
      (rawData.renewals && rawData.renewals.length > 0) ||
      (rawData.data && rawData.data.length > 0),
    [rawData]
  );

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  const handleKeyDown = useMemo(
    () => createActivationHandler(toggleExpanded),
    [toggleExpanded]
  );

  // Build list of present data types with icons (must be before early return for hooks rules)
  const presentDataTypes = useMemo(() => {
    const types: { key: string; icon: string; count: number }[] = [];
    if (rawData.companies?.length) types.push({ key: "companies", icon: "🏢", count: rawData.companies.length });
    if (rawData.activities?.length) types.push({ key: "activities", icon: "📋", count: rawData.activities.length });
    if (rawData.opportunities?.length) types.push({ key: "opportunities", icon: "💰", count: rawData.opportunities.length });
    if (rawData.history?.length) types.push({ key: "history", icon: "📜", count: rawData.history.length });
    if (rawData.renewals?.length) types.push({ key: "renewals", icon: "🔄", count: rawData.renewals.length });
    if (rawData.pipeline_summary) types.push({ key: "pipeline", icon: "📊", count: 1 });
    if (rawData.data?.length) types.push({ key: "records", icon: "📄", count: rawData.data.length });
    return types;
  }, [rawData]);

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
        <span className="data-section__preview" aria-label={`Contains ${presentDataTypes.map(t => `${t.count} ${t.key}`).join(", ")}`}>
          {presentDataTypes.map((type) => (
            <span key={type.key} className="data-section__preview-badge" title={`${type.key} (${type.count})`}>
              {type.count} {type.key}
            </span>
          ))}
        </span>
      </button>

      {expanded && (
        <div
          id="data-tables-content"
          className="data-section__content"
          role="region"
          aria-label="Data tables"
        >
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
            const HIDDEN_COLS = new Set(["notes", "closed_date", "created_at", "updated_at", "currency", "probability"]);
            const isVisible = (k: string) =>
              !k.startsWith("_") && !k.endsWith("_id") && !HIDDEN_COLS.has(k);

            // Check if data has a source column (from UNION ALL cross-table queries)
            const hasSourceColumn = "source" in rawData.data[0];

            if (hasSourceColumn) {
              // Group rows by source for separate tables with proper column labels
              const SOURCE_CONFIG: Record<string, { title: string; icon: string; columns: { key: string; label: string }[] }> = {
                company:     { title: "Company",      icon: "🏢", columns: [{ key: "name", label: "Name" }, { key: "plan", label: "Plan" }, { key: "status", label: "Status" }, { key: "health_status", label: "Health" }, { key: "key_date", label: "Renewal Date" }, { key: "notes", label: "Notes" }] },
                opportunity: { title: "Opportunities", icon: "💰", columns: [{ key: "name", label: "Name" }, { key: "plan", label: "Stage" }, { key: "status", label: "Type" }, { key: "health_status", label: "Value" }, { key: "key_date", label: "Close Date" }, { key: "notes", label: "Notes" }] },
                activity:    { title: "Activities",    icon: "📋", columns: [{ key: "name", label: "Type" }, { key: "plan", label: "Subject" }, { key: "status", label: "Status" }, { key: "health_status", label: "Priority" }, { key: "key_date", label: "Due Date" }, { key: "notes", label: "Notes" }] },
                history:     { title: "History",       icon: "📜", columns: [{ key: "name", label: "Type" }, { key: "plan", label: "Subject" }, { key: "key_date", label: "Date" }, { key: "notes", label: "Notes" }] },
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
                              {config.columns.map((col) => (
                                <th key={col.key} scope="col">{col.label}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map((row, idx) => (
                              <tr key={idx}>
                                {config.columns.map((col) => {
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
