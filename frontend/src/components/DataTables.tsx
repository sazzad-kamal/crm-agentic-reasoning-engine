import { useState } from "react";
import type { RawData } from "../types";

interface DataTablesProps {
  rawData: RawData;
}

/**
 * Collapsible data tables showing raw CRM data
 */
export function DataTables({ rawData }: DataTablesProps) {
  const [expanded, setExpanded] = useState(false);

  const hasData =
    (rawData.companies && rawData.companies.length > 0) ||
    (rawData.activities && rawData.activities.length > 0) ||
    (rawData.opportunities && rawData.opportunities.length > 0) ||
    (rawData.history && rawData.history.length > 0) ||
    (rawData.renewals && rawData.renewals.length > 0);

  if (!hasData) return null;

  return (
    <div className="data-section">
      <div
        className="data-section__header"
        onClick={() => setExpanded(!expanded)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && setExpanded(!expanded)}
      >
        <span>{expanded ? "▼" : "▶"}</span>
        <span>Data used</span>
      </div>

      {expanded && (
        <div className="data-section__content">
          {/* Companies Table */}
          {rawData.companies && rawData.companies.length > 0 && (
            <>
              <div className="data-table__title">Companies</div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Plan</th>
                    <th>Renewal Date</th>
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
            </>
          )}

          {/* Activities Table */}
          {rawData.activities && rawData.activities.length > 0 && (
            <>
              <div className="data-table__title">Activities</div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Type</th>
                    <th>Occurred At</th>
                    <th>Owner</th>
                    <th>Summary</th>
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
            </>
          )}

          {/* Opportunities Table */}
          {rawData.opportunities && rawData.opportunities.length > 0 && (
            <>
              <div className="data-table__title">Opportunities</div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Stage</th>
                    <th>Expected Close</th>
                    <th>Value</th>
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
            </>
          )}

          {/* History Table */}
          {rawData.history && rawData.history.length > 0 && (
            <>
              <div className="data-table__title">History</div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Event</th>
                    <th>Occurred At</th>
                    <th>Description</th>
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
            </>
          )}

          {/* Renewals Table */}
          {rawData.renewals && rawData.renewals.length > 0 && (
            <>
              <div className="data-table__title">Upcoming Renewals</div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Company</th>
                    <th>Plan</th>
                    <th>Renewal Date</th>
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
            </>
          )}

          {/* Pipeline Summary */}
          {rawData.pipeline_summary && (
            <>
              <div className="data-table__title">Pipeline Summary</div>
              <table className="data-table">
                <tbody>
                  <tr>
                    <td>Total Value</td>
                    <td>{formatCurrency(rawData.pipeline_summary.total_value)}</td>
                  </tr>
                  <tr>
                    <td>Opportunity Count</td>
                    <td>{rawData.pipeline_summary.count}</td>
                  </tr>
                </tbody>
              </table>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Utility functions for formatting
function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString();
  } catch {
    return dateStr;
  }
}

function formatDateTime(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleString();
  } catch {
    return dateStr;
  }
}

function formatCurrency(value: number): string {
  return `$${value.toLocaleString()}`;
}
