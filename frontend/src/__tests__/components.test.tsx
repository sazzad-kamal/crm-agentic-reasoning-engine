import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ErrorBanner } from "../components/ErrorBanner";
import { InputBar } from "../components/InputBar";
import { SourceChip, SourcesRow } from "../components/SourceChip";
import { StepPill, StepsRow } from "../components/StepPill";
import { FollowUpSuggestions } from "../components/FollowUpSuggestions";
import { MetaInfo } from "../components/MetaInfo";
import { LoadingDots, LoadingState } from "../components/LoadingDots";
import type { Source, Step } from "../types";

describe("ErrorBanner", () => {
  it("renders error message", () => {
    render(<ErrorBanner message="Test error" onDismiss={() => {}} />);
    expect(screen.getByText("Test error")).toBeInTheDocument();
  });

  it("calls onDismiss when close button clicked", () => {
    const onDismiss = vi.fn();
    render(<ErrorBanner message="Test error" onDismiss={onDismiss} />);
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("has correct ARIA attributes", () => {
    render(<ErrorBanner message="Test error" onDismiss={() => {}} />);
    const alert = screen.getByRole("alert");
    expect(alert).toBeInTheDocument();
    expect(alert).toHaveAttribute("aria-live", "assertive");
  });
});

describe("InputBar", () => {
  it("renders input with placeholder", () => {
    render(
      <InputBar value="" onChange={() => {}} onSubmit={() => {}} isLoading={false} />
    );
    expect(screen.getByPlaceholderText(/Ask a question/i)).toBeInTheDocument();
  });

  it("disables input when loading", () => {
    render(
      <InputBar value="" onChange={() => {}} onSubmit={() => {}} isLoading={true} />
    );
    expect(screen.getByPlaceholderText(/Ask a question/i)).toBeDisabled();
  });

  it("calls onChange when typing", () => {
    const onChange = vi.fn();
    render(
      <InputBar value="" onChange={onChange} onSubmit={() => {}} isLoading={false} />
    );
    fireEvent.change(screen.getByPlaceholderText(/Ask a question/i), {
      target: { value: "test" },
    });
    expect(onChange).toHaveBeenCalledWith("test");
  });

  it("calls onSubmit when form submitted", () => {
    const onSubmit = vi.fn();
    render(
      <InputBar value="test" onChange={() => {}} onSubmit={onSubmit} isLoading={false} />
    );
    fireEvent.submit(screen.getByPlaceholderText(/Ask a question/i).closest("form")!);
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("shows loading dots when loading", () => {
    render(
      <InputBar value="test" onChange={() => {}} onSubmit={() => {}} isLoading={true} />
    );
    expect(screen.getByText(/Thinking/i)).toBeInTheDocument();
  });
});

describe("SourceChip", () => {
  const source: Source = { type: "company", id: "test-1", label: "Acme Corp" };

  it("renders source label", () => {
    render(<SourceChip source={source} />);
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  it("renders correct icon for company type", () => {
    render(<SourceChip source={source} />);
    expect(screen.getByText("🏢")).toBeInTheDocument();
  });

  it("renders correct icon for doc type", () => {
    const docSource: Source = { type: "doc", id: "doc-1", label: "Help Doc" };
    render(<SourceChip source={docSource} />);
    expect(screen.getByText("📄")).toBeInTheDocument();
  });
});

describe("SourcesRow", () => {
  const sources: Source[] = [
    { type: "company", id: "1", label: "Company A" },
    { type: "doc", id: "2", label: "Doc B" },
  ];

  it("renders collapsed by default with source count", () => {
    render(<SourcesRow sources={sources} />);
    expect(screen.getByRole("button", { name: /sources/i })).toBeInTheDocument();
    expect(screen.getByText(/Sources \(2\)/)).toBeInTheDocument();
  });

  it("expands to show sources when clicked", () => {
    render(<SourcesRow sources={sources} />);
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByText("Company A")).toBeInTheDocument();
    expect(screen.getByText("Doc B")).toBeInTheDocument();
  });
});

describe("StepPill", () => {
  const step: Step = { id: "step-1", label: "Processing", status: "done" };

  it("renders step label", () => {
    render(<StepPill step={step} />);
    expect(screen.getByText(/Processing/)).toBeInTheDocument();
  });

  it("applies correct class for done status", () => {
    const { container } = render(<StepPill step={step} />);
    expect(container.querySelector(".step-pill--done")).toBeInTheDocument();
  });

  it("applies correct class for error status", () => {
    const errorStep: Step = { id: "step-1", label: "Failed", status: "error" };
    const { container } = render(<StepPill step={errorStep} />);
    expect(container.querySelector(".step-pill--error")).toBeInTheDocument();
  });
});

describe("StepsRow", () => {
  const steps: Step[] = [
    { id: "1", label: "Step 1", status: "done" },
    { id: "2", label: "Step 2", status: "running" },
  ];

  it("renders all steps", () => {
    render(<StepsRow steps={steps} />);
    expect(screen.getByText(/Step 1/)).toBeInTheDocument();
    expect(screen.getByText(/Step 2/)).toBeInTheDocument();
  });
});

describe("FollowUpSuggestions", () => {
  const suggestions = ["Question 1?", "Question 2?"];

  it("renders all suggestions", () => {
    render(<FollowUpSuggestions suggestions={suggestions} onSuggestionClick={() => {}} />);
    expect(screen.getByText("Question 1?")).toBeInTheDocument();
    expect(screen.getByText("Question 2?")).toBeInTheDocument();
  });

  it("renders Ask label", () => {
    render(<FollowUpSuggestions suggestions={suggestions} onSuggestionClick={() => {}} />);
    expect(screen.getByText("Ask:")).toBeInTheDocument();
  });

  it("calls onSuggestionClick when clicked", () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions suggestions={suggestions} onSuggestionClick={onClick} />);
    fireEvent.click(screen.getByText("Question 1?"));
    expect(onClick).toHaveBeenCalledWith("Question 1?");
  });

  it("supports keyboard activation with Enter", () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions suggestions={suggestions} onSuggestionClick={onClick} />);
    const button = screen.getByText("Question 1?");
    fireEvent.keyDown(button, { key: "Enter" });
    expect(onClick).toHaveBeenCalledWith("Question 1?");
  });

  it("supports keyboard activation with Space", () => {
    const onClick = vi.fn();
    render(<FollowUpSuggestions suggestions={suggestions} onSuggestionClick={onClick} />);
    const button = screen.getByText("Question 2?");
    fireEvent.keyDown(button, { key: " " });
    expect(onClick).toHaveBeenCalledWith("Question 2?");
  });

  it("has correct ARIA attributes", () => {
    render(<FollowUpSuggestions suggestions={suggestions} onSuggestionClick={() => {}} />);
    expect(screen.getByRole("group", { name: /suggested follow-up/i })).toBeInTheDocument();
  });
});

describe("MetaInfo", () => {
  it("renders latency info", () => {
    render(<MetaInfo meta={{ mode_used: "data", latency_ms: 150 }} />);
    expect(screen.getByText(/150ms/)).toBeInTheDocument();
  });

  it("renders mode info", () => {
    render(<MetaInfo meta={{ mode_used: "data+docs", latency_ms: 100 }} />);
    expect(screen.getByText(/data\+docs/)).toBeInTheDocument();
  });
});

describe("LoadingDots", () => {
  it("renders loading dots", () => {
    const { container } = render(<LoadingDots />);
    expect(container.querySelectorAll(".loading-dot")).toHaveLength(3);
  });
});

describe("LoadingState", () => {
  it("renders loading message", () => {
    render(<LoadingState />);
    expect(screen.getByText(/Thinking/i)).toBeInTheDocument();
  });

  it("has correct ARIA attributes", () => {
    render(<LoadingState />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});
