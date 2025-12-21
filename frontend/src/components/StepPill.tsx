import type { Step } from "../types";

const STATUS_ICONS: Record<string, string> = {
  done: "✓",
  running: "⟳",
  pending: "○",
  error: "✕",
  skipped: "–",
};

interface StepPillProps {
  step: Step;
}

/**
 * Displays a single step as a styled pill
 */
export function StepPill({ step }: StepPillProps) {
  const icon = STATUS_ICONS[step.status] || "○";

  return (
    <span className={`step-pill step-pill--${step.status}`}>
      {icon} {step.label}
    </span>
  );
}

interface StepsRowProps {
  steps: Step[];
}

/**
 * Displays a row of step pills
 */
export function StepsRow({ steps }: StepsRowProps) {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="steps-row">
      <span className="steps-row__label">Steps:</span>
      {steps.map((step) => (
        <StepPill key={step.id} step={step} />
      ))}
    </div>
  );
}
