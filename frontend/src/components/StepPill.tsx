import { memo } from "react";
import type { Step } from "../types";

/** Icons for different step statuses */
const STATUS_ICONS: Record<Step["status"], string> = {
  done: "✓",
  running: "⟳",
  pending: "○",
  error: "✕",
  skipped: "–",
};

/** Human-readable labels for step statuses */
const STATUS_LABELS: Record<Step["status"], string> = {
  done: "Completed",
  running: "In progress",
  pending: "Pending",
  error: "Failed",
  skipped: "Skipped",
};

interface StepPillProps {
  step: Step;
  index?: number;  // For staggered animation
}

/**
 * Displays a single step as a styled pill with status indicator.
 * Memoized for performance in step lists.
 */
export const StepPill = memo(function StepPill({ step, index = 0 }: StepPillProps) {
  const icon = STATUS_ICONS[step.status] || "○";
  const statusLabel = STATUS_LABELS[step.status] || step.status;
  
  // Stagger animation delay based on index
  const animationDelay = `${index * 100}ms`;

  return (
    <span
      className={`chip step-pill step-pill--${step.status}`}
      role="listitem"
      aria-label={`${step.label}: ${statusLabel}`}
      data-status={step.status}
      style={{ animationDelay }}
    >
      <span className="step-pill__icon" aria-hidden="true">{icon}</span> {step.label}
    </span>
  );
});

interface StepsRowProps {
  steps: Step[];
}

/**
 * Displays a row of step pills showing workflow progress.
 * Memoized to prevent re-renders when steps haven't changed.
 */
export const StepsRow = memo(function StepsRow({ steps }: StepsRowProps) {
  if (!steps || steps.length === 0) return null;

  const completedCount = steps.filter((s) => s.status === "done").length;

  return (
    <div
      className="flex-row steps-row"
      role="list"
      aria-label={`Workflow steps: ${completedCount} of ${steps.length} completed`}
    >
      <span className="steps-row__label" aria-hidden="true">
        Steps:
      </span>
      {steps.map((step, index) => (
        <StepPill key={step.id} step={step} index={index} />
      ))}
    </div>
  );
});
