import type { Source } from "../types";

const SOURCE_ICONS: Record<string, string> = {
  company: "🏢",
  doc: "📄",
  activity: "📋",
  opportunity: "💰",
  history: "📜",
};

interface SourceChipProps {
  source: Source;
}

/**
 * Displays a single source as a styled chip
 */
export function SourceChip({ source }: SourceChipProps) {
  const icon = SOURCE_ICONS[source.type] || "📌";

  return (
    <span className="source-chip">
      <span className="source-chip__icon">{icon}</span>
      {source.label}
    </span>
  );
}

interface SourcesRowProps {
  sources: Source[];
}

/**
 * Displays a row of source chips
 */
export function SourcesRow({ sources }: SourcesRowProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-row">
      <span className="sources-row__label">Sources:</span>
      {sources.map((source) => (
        <SourceChip key={source.id} source={source} />
      ))}
    </div>
  );
}
