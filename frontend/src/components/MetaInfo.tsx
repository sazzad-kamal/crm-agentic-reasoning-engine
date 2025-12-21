import type { Meta } from "../types";

interface MetaInfoProps {
  meta: Meta;
}

/**
 * Displays response metadata (latency, mode, etc.)
 */
export function MetaInfo({ meta }: MetaInfoProps) {
  const parts: string[] = [];

  if (meta.latency_ms) {
    parts.push(`${meta.latency_ms}ms`);
  }

  if (meta.mode_used) {
    parts.push(`Mode: ${meta.mode_used}`);
  }

  if (meta.company_id) {
    parts.push(`Company: ${meta.company_id}`);
  }

  if (parts.length === 0) return null;

  return <div className="meta-line">{parts.join(" · ")}</div>;
}
