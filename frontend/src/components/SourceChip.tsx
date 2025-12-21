import { memo } from "react";
import type { Source } from "../types";

/** Icons for different source types */
const SOURCE_ICONS: Record<Source["type"], string> = {
  company: "🏢",
  doc: "📄",
  activity: "📋",
  opportunity: "💰",
  history: "📜",
};

/** Human-readable labels for source types */
const SOURCE_TYPE_LABELS: Record<Source["type"], string> = {
  company: "Company",
  doc: "Document",
  activity: "Activity",
  opportunity: "Opportunity",
  history: "History",
};

interface SourceChipProps {
  source: Source;
}

/**
 * Displays a single source as a styled chip.
 * Memoized for performance in lists.
 */
export const SourceChip = memo(function SourceChip({ source }: SourceChipProps) {
  const icon = SOURCE_ICONS[source.type] || "📌";
  const typeLabel = SOURCE_TYPE_LABELS[source.type] || source.type;

  return (
    <span
      className="source-chip"
      role="listitem"
      aria-label={`${typeLabel}: ${source.label}`}
    >
      <span className="source-chip__icon" aria-hidden="true">
        {icon}
      </span>
      <span className="source-chip__label">{source.label}</span>
    </span>
  );
});

interface SourcesRowProps {
  sources: Source[];
}

/**
 * Displays a row of source chips.
 * Memoized to prevent re-renders when sources haven't changed.
 */
export const SourcesRow = memo(function SourcesRow({ sources }: SourcesRowProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <div
      className="sources-row"
      role="list"
      aria-label={`${sources.length} source${sources.length !== 1 ? "s" : ""} referenced`}
    >
      <span className="sources-row__label" aria-hidden="true">
        Sources:
      </span>
      {sources.map((source) => (
        <SourceChip key={source.id} source={source} />
      ))}
    </div>
  );
});
