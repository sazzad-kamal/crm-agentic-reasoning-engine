import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StepPill, StepsRow } from "../components/StepPill";
import type { Step } from "../types";

describe("StepPill", () => {
  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders done step with correct icon and label", () => {
    const step: Step = {
      id: "step1",
      label: "Data fetched",
      status: "done",
    };

    render(<StepPill step={step} />);

    expect(screen.getByText("Data fetched")).toBeInTheDocument();
    expect(screen.getByText("✓")).toBeInTheDocument();
  });

  it("renders running step with correct icon", () => {
    const step: Step = {
      id: "step2",
      label: "Processing",
      status: "running",
    };

    render(<StepPill step={step} />);

    expect(screen.getByText("Processing")).toBeInTheDocument();
    expect(screen.getByText("⟳")).toBeInTheDocument();
  });

  it("renders pending step with correct icon", () => {
    const step: Step = {
      id: "step3",
      label: "Waiting",
      status: "pending",
    };

    render(<StepPill step={step} />);

    expect(screen.getByText("Waiting")).toBeInTheDocument();
    expect(screen.getByText("○")).toBeInTheDocument();
  });

  it("renders error step with correct icon", () => {
    const step: Step = {
      id: "step4",
      label: "Failed",
      status: "error",
    };

    render(<StepPill step={step} />);

    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("✕")).toBeInTheDocument();
  });

  it("renders skipped step with correct icon", () => {
    const step: Step = {
      id: "step5",
      label: "Skipped",
      status: "skipped",
    };

    render(<StepPill step={step} />);

    expect(screen.getByText("Skipped")).toBeInTheDocument();
    expect(screen.getByText("–")).toBeInTheDocument();
  });

  // =========================================================================
  // CSS Classes and Styling
  // =========================================================================

  it("applies correct CSS class for done status", () => {
    const step: Step = {
      id: "step1",
      label: "Done",
      status: "done",
    };

    const { container } = render(<StepPill step={step} />);
    const pill = container.querySelector(".step-pill--done");

    expect(pill).toBeInTheDocument();
  });

  it("applies correct CSS class for running status", () => {
    const step: Step = {
      id: "step2",
      label: "Running",
      status: "running",
    };

    const { container } = render(<StepPill step={step} />);
    const pill = container.querySelector(".step-pill--running");

    expect(pill).toBeInTheDocument();
  });

  it("applies correct CSS class for error status", () => {
    const step: Step = {
      id: "step3",
      label: "Error",
      status: "error",
    };

    const { container } = render(<StepPill step={step} />);
    const pill = container.querySelector(".step-pill--error");

    expect(pill).toBeInTheDocument();
  });

  // =========================================================================
  // Animation Delay
  // =========================================================================

  it("applies animation delay based on index", () => {
    const step: Step = {
      id: "step1",
      label: "Step",
      status: "done",
    };

    const { container } = render(<StepPill step={step} index={3} />);
    const pill = container.querySelector(".step-pill");

    expect(pill).toHaveStyle({ animationDelay: "300ms" });
  });

  it("uses default index 0 when not provided", () => {
    const step: Step = {
      id: "step1",
      label: "Step",
      status: "done",
    };

    const { container } = render(<StepPill step={step} />);
    const pill = container.querySelector(".step-pill");

    expect(pill).toHaveStyle({ animationDelay: "0ms" });
  });

  it("calculates delay correctly for various indices", () => {
    const step: Step = {
      id: "step1",
      label: "Step",
      status: "done",
    };

    // Test index 0
    const { container: c1 } = render(<StepPill step={step} index={0} />);
    expect(c1.querySelector(".step-pill")).toHaveStyle({ animationDelay: "0ms" });

    // Test index 5
    const { container: c2 } = render(<StepPill step={step} index={5} />);
    expect(c2.querySelector(".step-pill")).toHaveStyle({ animationDelay: "500ms" });

    // Test index 10
    const { container: c3 } = render(<StepPill step={step} index={10} />);
    expect(c3.querySelector(".step-pill")).toHaveStyle({ animationDelay: "1000ms" });
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("has proper ARIA role", () => {
    const step: Step = {
      id: "step1",
      label: "Data fetched",
      status: "done",
    };

    render(<StepPill step={step} />);

    const pill = screen.getByRole("listitem");
    expect(pill).toBeInTheDocument();
  });

  it("has descriptive ARIA label for done step", () => {
    const step: Step = {
      id: "step1",
      label: "Data fetched",
      status: "done",
    };

    render(<StepPill step={step} />);

    expect(screen.getByLabelText("Data fetched: Completed")).toBeInTheDocument();
  });

  it("has descriptive ARIA label for running step", () => {
    const step: Step = {
      id: "step2",
      label: "Processing",
      status: "running",
    };

    render(<StepPill step={step} />);

    expect(screen.getByLabelText("Processing: In progress")).toBeInTheDocument();
  });

  it("has descriptive ARIA label for error step", () => {
    const step: Step = {
      id: "step3",
      label: "Validation",
      status: "error",
    };

    render(<StepPill step={step} />);

    expect(screen.getByLabelText("Validation: Failed")).toBeInTheDocument();
  });

  it("hides icon from screen readers", () => {
    const step: Step = {
      id: "step1",
      label: "Step",
      status: "done",
    };

    render(<StepPill step={step} />);
    const icon = screen.getByText("✓");

    expect(icon).toHaveAttribute("aria-hidden", "true");
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles unknown status with fallback icon", () => {
    const step = {
      id: "step1",
      label: "Custom step",
      status: "custom" as Step["status"],
    };

    render(<StepPill step={step} />);

    expect(screen.getByText("Custom step")).toBeInTheDocument();
    expect(screen.getByText("○")).toBeInTheDocument();
  });

  it("handles empty label", () => {
    const step: Step = {
      id: "step1",
      label: "",
      status: "done",
    };

    render(<StepPill step={step} />);

    const pill = screen.getByRole("listitem");
    expect(pill).toBeInTheDocument();
  });

  it("handles very long label", () => {
    const step: Step = {
      id: "step1",
      label: "A".repeat(200),
      status: "done",
    };

    render(<StepPill step={step} />);

    expect(screen.getByText("A".repeat(200))).toBeInTheDocument();
  });

  it("handles special characters in label", () => {
    const step: Step = {
      id: "step1",
      label: "Step <1> & \"Test\"",
      status: "done",
    };

    render(<StepPill step={step} />);

    expect(screen.getByText('Step <1> & "Test"')).toBeInTheDocument();
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("re-renders when step changes", () => {
    const step1: Step = {
      id: "step1",
      label: "First",
      status: "done",
    };

    const step2: Step = {
      id: "step2",
      label: "Second",
      status: "running",
    };

    const { rerender } = render(<StepPill step={step1} />);
    expect(screen.getByText("First")).toBeInTheDocument();

    rerender(<StepPill step={step2} />);
    expect(screen.queryByText("First")).not.toBeInTheDocument();
    expect(screen.getByText("Second")).toBeInTheDocument();
  });
});

describe("StepsRow", () => {
  const mockSteps: Step[] = [
    { id: "step1", label: "Route question", status: "done" },
    { id: "step2", label: "Fetch data", status: "done" },
    { id: "step3", label: "Generate answer", status: "running" },
    { id: "step4", label: "Format response", status: "pending" },
  ];

  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders all steps", () => {
    render(<StepsRow steps={mockSteps} />);

    expect(screen.getByText("Route question")).toBeInTheDocument();
    expect(screen.getByText("Fetch data")).toBeInTheDocument();
    expect(screen.getByText("Generate answer")).toBeInTheDocument();
    expect(screen.getByText("Format response")).toBeInTheDocument();
  });

  it("renders steps label", () => {
    render(<StepsRow steps={mockSteps} />);

    expect(screen.getByText("Steps:")).toBeInTheDocument();
  });

  it("renders correct number of step pills", () => {
    render(<StepsRow steps={mockSteps} />);

    const pills = screen.getAllByRole("listitem");
    expect(pills).toHaveLength(4);
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("has proper ARIA role", () => {
    render(<StepsRow steps={mockSteps} />);

    const list = screen.getByRole("list");
    expect(list).toBeInTheDocument();
  });

  it("has descriptive ARIA label with progress", () => {
    render(<StepsRow steps={mockSteps} />);

    // 2 done out of 4 total
    expect(screen.getByLabelText("Workflow steps: 2 of 4 completed")).toBeInTheDocument();
  });

  it("updates ARIA label when all steps complete", () => {
    const allDone: Step[] = [
      { id: "step1", label: "Step 1", status: "done" },
      { id: "step2", label: "Step 2", status: "done" },
      { id: "step3", label: "Step 3", status: "done" },
    ];

    render(<StepsRow steps={allDone} />);

    expect(screen.getByLabelText("Workflow steps: 3 of 3 completed")).toBeInTheDocument();
  });

  it("shows 0 completed when no steps are done", () => {
    const noneComplete: Step[] = [
      { id: "step1", label: "Step 1", status: "pending" },
      { id: "step2", label: "Step 2", status: "pending" },
    ];

    render(<StepsRow steps={noneComplete} />);

    expect(screen.getByLabelText("Workflow steps: 0 of 2 completed")).toBeInTheDocument();
  });

  it("hides steps label from screen readers", () => {
    render(<StepsRow steps={mockSteps} />);

    const label = screen.getByText("Steps:");
    expect(label).toHaveAttribute("aria-hidden", "true");
  });

  // =========================================================================
  // Step Ordering and Animation
  // =========================================================================

  it("passes correct index to each step pill", () => {
    const { container } = render(<StepsRow steps={mockSteps} />);
    const pills = container.querySelectorAll(".step-pill");

    expect(pills[0]).toHaveStyle({ animationDelay: "0ms" });
    expect(pills[1]).toHaveStyle({ animationDelay: "100ms" });
    expect(pills[2]).toHaveStyle({ animationDelay: "200ms" });
    expect(pills[3]).toHaveStyle({ animationDelay: "300ms" });
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("returns null for empty steps array", () => {
    const { container } = render(<StepsRow steps={[]} />);

    expect(container.firstChild).toBeNull();
  });

  it("handles single step", () => {
    const singleStep: Step[] = [
      { id: "step1", label: "Only step", status: "done" },
    ];

    render(<StepsRow steps={singleStep} />);

    expect(screen.getByText("Only step")).toBeInTheDocument();
    expect(screen.getByLabelText("Workflow steps: 1 of 1 completed")).toBeInTheDocument();
  });

  it("handles many steps", () => {
    const manySteps: Step[] = Array.from({ length: 20 }, (_, i) => ({
      id: `step${i}`,
      label: `Step ${i}`,
      status: i < 10 ? ("done" as const) : ("pending" as const),
    }));

    render(<StepsRow steps={manySteps} />);

    expect(screen.getByText("Step 0")).toBeInTheDocument();
    expect(screen.getByText("Step 19")).toBeInTheDocument();
    expect(screen.getByLabelText("Workflow steps: 10 of 20 completed")).toBeInTheDocument();
  });

  it("handles all error steps", () => {
    const errorSteps: Step[] = [
      { id: "step1", label: "Failed 1", status: "error" },
      { id: "step2", label: "Failed 2", status: "error" },
    ];

    render(<StepsRow steps={errorSteps} />);

    expect(screen.getByLabelText("Workflow steps: 0 of 2 completed")).toBeInTheDocument();
    expect(screen.getByText("Failed 1")).toBeInTheDocument();
    expect(screen.getByText("Failed 2")).toBeInTheDocument();
  });

  it("handles mixed status steps", () => {
    const mixedSteps: Step[] = [
      { id: "step1", label: "Done", status: "done" },
      { id: "step2", label: "Running", status: "running" },
      { id: "step3", label: "Error", status: "error" },
      { id: "step4", label: "Skipped", status: "skipped" },
      { id: "step5", label: "Pending", status: "pending" },
    ];

    render(<StepsRow steps={mixedSteps} />);

    // Only 1 done out of 5
    expect(screen.getByLabelText("Workflow steps: 1 of 5 completed")).toBeInTheDocument();
  });

  it("handles duplicate step IDs", () => {
    const duplicateSteps: Step[] = [
      { id: "step1", label: "First", status: "done" },
      { id: "step1", label: "Second", status: "running" },
    ];

    render(<StepsRow steps={duplicateSteps} />);

    expect(screen.getByText("First")).toBeInTheDocument();
    expect(screen.getByText("Second")).toBeInTheDocument();
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("re-renders when steps change", () => {
    const steps1: Step[] = [
      { id: "step1", label: "First", status: "done" },
    ];

    const steps2: Step[] = [
      { id: "step1", label: "First", status: "done" },
      { id: "step2", label: "Second", status: "running" },
    ];

    const { rerender } = render(<StepsRow steps={steps1} />);
    expect(screen.getByLabelText("Workflow steps: 1 of 1 completed")).toBeInTheDocument();

    rerender(<StepsRow steps={steps2} />);
    expect(screen.getByLabelText("Workflow steps: 1 of 2 completed")).toBeInTheDocument();
    expect(screen.getByText("Second")).toBeInTheDocument();
  });
});
