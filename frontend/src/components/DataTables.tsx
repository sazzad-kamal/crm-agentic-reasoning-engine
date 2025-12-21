import { useState, useCallback, useMemo, memo } from "react";
import type { RawData } from "../types";

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

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleExpanded();
      }
    },
    [toggleExpanded]
  );

  if (!hasData) return null;

  const tableCount = [
    rawData.companies?.length,
    rawData.activities?.length,
    rawData.opportunities?.length,
    rawData.history?.length,
    rawData.renewals?.length,
  ].filter(Boolean).length;

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
        <span aria-hidden="true">{expanded ? "▼" : "▶"}</span>
        <span>Data used ({tableCount} table{tableCount !== 1 ? "s" : ""})</span>
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
            <div role="region" aria-label="Companies data">
              <h4 className="data-table__title" id="companies-table-label">
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
            <div role="region" aria-label="Activities data">
              <h4 className="data-table__title" id="activities-table-label">
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
            <div role="region" aria-label="Opportunities data">
              <h4 className="data-table__title" id="opportunities-table-label">
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
            <div role="region" aria-label="History data">
              <h4 className="data-table__title" id="history-table-label">
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
            <div role="region" aria-label="Renewals data">
              <h4 className="data-table__title" id="renewals-table-label">
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
            <div role="region" aria-label="Pipeline summary data">
              <h4 className="data-table__title" id="pipeline-table-label">
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

// Utility functions for formatting
function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return dateStr;
  }
}

function formatDateTime(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateStr;
  }
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}
