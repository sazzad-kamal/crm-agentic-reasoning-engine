import { useState, useCallback, useMemo, memo } from "react";
import type { KeyboardEvent } from "react";
import type { RawData } from "../types";
import { formatDate, formatDateTime, formatCurrency } from "../utils/formatters";

/** Nested data item display (simplified version of NestedDataDisplay) */
function NestedItems({ items, type }: { items: unknown[]; type: "private_texts" | "attachments" }) {
  if (!items || items.length === 0) return null;

  return (
    <div className="nested-items">
      {items.slice(0, 5).map((item, idx) => {
        const record = item as Record<string, unknown>;
        if (type === "private_texts") {
          return (
            <div key={idx} className="nested-items__item">
              <span className="nested-items__type">{String(record.type || "note")}</span>
              <span className="nested-items__text">{String(record.title || record.text || "").slice(0, 100)}</span>
            </div>
          );
        }
        // attachments
        return (
          <div key={idx} className="nested-items__item">
            <span className="nested-items__icon">📎</span>
            <span className="nested-items__text">{String(record.title || record.file_name || "Attachment")}</span>
            <span className="nested-items__meta">{String(record.file_type || "")}</span>
          </div>
        );
      })}
      {items.length > 5 && (
        <div className="nested-items__more">+{items.length - 5} more</div>
      )}
    </div>
  );
}

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
      (rawData.renewals && rawData.renewals.length > 0),
    [rawData]
  );

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  const handleKeyDown = useMemo(
    () => createActivationHandler(toggleExpanded),
    [toggleExpanded]
  );

  if (!hasData) return null;

  // Build list of present data types with icons
  const presentDataTypes = useMemo(() => {
    const types: { key: string; icon: string; count: number }[] = [];
    if (rawData.companies?.length) types.push({ key: "companies", icon: "🏢", count: rawData.companies.length });
    if (rawData.activities?.length) types.push({ key: "activities", icon: "📋", count: rawData.activities.length });
    if (rawData.opportunities?.length) types.push({ key: "opportunities", icon: "💰", count: rawData.opportunities.length });
    if (rawData.history?.length) types.push({ key: "history", icon: "📜", count: rawData.history.length });
    if (rawData.renewals?.length) types.push({ key: "renewals", icon: "🔄", count: rawData.renewals.length });
    if (rawData.pipeline_summary) types.push({ key: "pipeline", icon: "📊", count: 1 });
    return types;
  }, [rawData]);

  const tableCount = presentDataTypes.length;

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
        <span className="data-section__arrow" aria-hidden="true">▶</span>
        <span className="data-section__label">Data used</span>
        <span className="data-section__preview" aria-label={`Contains ${presentDataTypes.map(t => t.key).join(", ")}`}>
          {presentDataTypes.map((type) => (
            <span key={type.key} className="data-section__preview-icon" title={`${type.key} (${type.count})`}>
              {type.icon}
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
                    <th scope="col">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.companies.map((c) => (
                    <tr key={c.company_id}>
                      <td>{c.name}</td>
                      <td>{c.plan}</td>
                      <td>{formatDate(c.renewal_date)}</td>
                      <td>
                        <NestedItems items={c._private_texts || []} type="private_texts" />
                      </td>
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
                    <th scope="col">Attachments</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.opportunities.map((o) => (
                    <tr key={o.opportunity_id}>
                      <td>{o.name}</td>
                      <td>{o.stage}</td>
                      <td>{formatDate(o.expected_close_date)}</td>
                      <td>{formatCurrency(o.value)}</td>
                      <td>
                        <NestedItems items={o._attachments || []} type="attachments" />
                      </td>
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
                    <th scope="col">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {rawData.history.map((h) => (
                    <tr key={h.history_id}>
                      <td>{h.event_type}</td>
                      <td>{formatDateTime(h.occurred_at)}</td>
                      <td>{h.description || "—"}</td>
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
        </div>
      )}
    </section>
  );
});
