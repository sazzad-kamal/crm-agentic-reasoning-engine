import { memo } from "react";
import type { Meta } from "../types";

interface MetaInfoProps {
  meta: Meta;
}

/**
 * Displays response metadata (latency, mode, etc.).
 * Memoized for performance.
 */
export const MetaInfo = memo(function MetaInfo({ meta }: MetaInfoProps) {
  const parts: string[] = [];

  if (meta.latency_ms !== undefined) {
    parts.push(`${meta.latency_ms}ms`);
  }

  if (meta.mode_used) {
    parts.push(`Mode: ${meta.mode_used}`);
  }

  if (meta.company_id) {
    parts.push(`Company: ${meta.company_id}`);
  }

  if (parts.length === 0) return null;

  return (
    <div
      className="meta-line"
      role="contentinfo"
      aria-label={`Response metadata: ${parts.join(", ")}`}
    >
      {parts.join(" · ")}
    </div>
  );
});
