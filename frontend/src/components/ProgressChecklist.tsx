import type { FetchStep } from "../types";

// User-friendly labels for each step ID
const STEP_LABELS: Record<string, string> = {
  calendar: "Checking calendar",
  contacts: "Loading contacts",
  history: "Reviewing history",
  opportunities: "Analyzing opportunities",
  companies: "Loading companies",
};

interface ProgressChecklistProps {
  steps: FetchStep[];
}

/**
 * Displays a checklist of fetch steps with their current status.
 * Shows all steps upfront (grayed when pending), marks each as done/error as they complete.
 */
export function ProgressChecklist({ steps }: ProgressChecklistProps) {
  if (!steps || steps.length === 0) {
    return null;
  }

  return (
    <div className="progress-checklist" aria-live="polite">
      {steps.map((step) => (
        <div
          key={step.id}
          className={`progress-checklist__step progress-checklist__step--${step.status}`}
        >
          <span className="progress-checklist__icon">
            {step.status === "done" && "\u2713"}
            {step.status === "error" && "\u2717"}
            {step.status === "cached" && "\u21BB"}
            {step.status === "pending" && "\u25CB"}
          </span>
          <span className="progress-checklist__label">
            {STEP_LABELS[step.id] || step.id}
            {step.status === "cached" && " (cached)"}
          </span>
        </div>
      ))}
    </div>
  );
}
