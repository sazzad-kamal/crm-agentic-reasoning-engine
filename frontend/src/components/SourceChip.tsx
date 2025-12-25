import { memo, useState, useCallback, useMemo } from "react";
import type { Source } from "../types";
import { createActivationHandler } from "../utils/keyboard";

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
      className="chip source-chip"
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
 * Displays a collapsible row of source chips.
 * Collapsed by default, click to expand.
 * Memoized to prevent re-renders when sources haven't changed.
 */
export const SourcesRow = memo(function SourcesRow({ sources }: SourcesRowProps) {
  const [expanded, setExpanded] = useState(false);

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  const handleKeyDown = useMemo(
    () => createActivationHandler(toggleExpanded),
    [toggleExpanded]
  );

  if (!sources || sources.length === 0) return null;

  return (
    <section className="sources-section" aria-label="Sources referenced">
      <button
        className="sources-section__header"
        onClick={toggleExpanded}
        onKeyDown={handleKeyDown}
        type="button"
        aria-expanded={expanded}
        aria-controls="sources-content"
      >
        <span aria-hidden="true">{expanded ? "▼" : "▶"}</span>
        <span>Sources ({sources.length})</span>
      </button>

      {expanded && (
        <div
          id="sources-content"
          className="sources-section__content"
          role="list"
          aria-label={`${sources.length} source${sources.length !== 1 ? "s" : ""} referenced`}
        >
          {sources.map((source) => (
            <SourceChip key={source.id} source={source} />
          ))}
        </div>
      )}
    </section>
  );
});
