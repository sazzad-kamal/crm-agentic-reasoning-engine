import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DataTables } from "../components/DataTables";
import { MessageBlock } from "../components/MessageBlock";
import { SkipLink } from "../components/SkipLink";
import { MessageSkeleton, ChatSkeleton } from "../components/Skeleton";
import type { ChatMessage, RawData } from "../types";

describe("DataTables", () => {
  const mockRawData: RawData = {
    companies: [
      { company_id: "1", name: "Acme Corp", plan: "Enterprise", renewal_date: "2025-06-15" },
    ],
    activities: [
      { activity_id: "1", type: "call", occurred_at: "2025-01-15T10:00:00Z", owner: "John", summary: "Follow-up call" },
    ],
    opportunities: [
      { opportunity_id: "1", name: "Big Deal", stage: "Proposal", expected_close_date: "2025-03-01", value: 50000 },
    ],
  };

  it("renders nothing when no data provided", () => {
    const { container } = render(<DataTables rawData={{}} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders collapsed by default", () => {
    render(<DataTables rawData={mockRawData} />);
    expect(screen.getByRole("button", { name: /data used/i })).toBeInTheDocument();
    expect(screen.queryByText("Acme Corp")).not.toBeInTheDocument();
  });

  it("expands when header clicked", () => {
    render(<DataTables rawData={mockRawData} />);
    fireEvent.click(screen.getByRole("button", { name: /data used/i }));
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  it("expands on Enter key press", () => {
    render(<DataTables rawData={mockRawData} />);
    const header = screen.getByRole("button", { name: /data used/i });
    fireEvent.keyDown(header, { key: "Enter" });
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  it("expands on Space key press", () => {
    render(<DataTables rawData={mockRawData} />);
    const header = screen.getByRole("button", { name: /data used/i });
    fireEvent.keyDown(header, { key: " " });
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
  });

  it("has correct aria-expanded attribute", () => {
    render(<DataTables rawData={mockRawData} />);
    const header = screen.getByRole("button", { name: /data used/i });
    expect(header).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(header);
    expect(header).toHaveAttribute("aria-expanded", "true");
  });

  it("shows companies table with proper accessibility", () => {
    render(<DataTables rawData={mockRawData} />);
    fireEvent.click(screen.getByRole("button", { name: /data used/i }));
    expect(screen.getByRole("region", { name: /companies data/i })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /companies/i })).toBeInTheDocument();
  });

  it("shows activities table with proper accessibility", () => {
    render(<DataTables rawData={mockRawData} />);
    fireEvent.click(screen.getByRole("button", { name: /data used/i }));
    expect(screen.getByRole("region", { name: /activities data/i })).toBeInTheDocument();
  });

  it("shows opportunities table with proper accessibility", () => {
    render(<DataTables rawData={mockRawData} />);
    fireEvent.click(screen.getByRole("button", { name: /data used/i }));
    expect(screen.getByRole("region", { name: /opportunities data/i })).toBeInTheDocument();
  });

  it("formats currency values correctly", () => {
    render(<DataTables rawData={mockRawData} />);
    fireEvent.click(screen.getByRole("button", { name: /data used/i }));
    expect(screen.getByText(/\$50,000/)).toBeInTheDocument();
  });

  it("displays table count in header", () => {
    render(<DataTables rawData={mockRawData} />);
    expect(screen.getByText(/3 tables/)).toBeInTheDocument();
  });
});

describe("MessageBlock", () => {
  const mockMessage: ChatMessage = {
    id: "msg-1",
    question: "What is happening with Acme?",
    response: {
      answer: "Acme is doing great!",
      sources: [{ type: "company", id: "1", label: "Acme Corp" }],
      steps: [{ id: "1", label: "Query", status: "done" }],
      meta: { mode_used: "data", latency_ms: 150 },
    },
    timestamp: new Date(),
  };

  it("renders user question", () => {
    render(<MessageBlock message={mockMessage} />);
    expect(screen.getByText("What is happening with Acme?")).toBeInTheDocument();
  });

  it("renders assistant answer", () => {
    render(<MessageBlock message={mockMessage} />);
    expect(screen.getByText("Acme is doing great!")).toBeInTheDocument();
  });

  it("has correct ARIA role and label", () => {
    render(<MessageBlock message={mockMessage} />);
    expect(screen.getByLabelText(/conversation about/i)).toBeInTheDocument();
  });

  it("shows thinking indicator when no response", () => {
    const loadingMessage: ChatMessage = {
      id: "msg-2",
      question: "Loading question",
      response: null,
      timestamp: new Date(),
    };
    render(<MessageBlock message={loadingMessage} />);
    expect(screen.getByRole("status", { name: /thinking/i })).toBeInTheDocument();
  });

  it("calls onFollowUpClick when follow-up clicked", () => {
    const messageWithFollowUp: ChatMessage = {
      ...mockMessage,
      response: {
        ...mockMessage.response!,
        follow_up_suggestions: ["Tell me more about Acme"],
      },
    };
    const onClick = vi.fn();
    render(<MessageBlock message={messageWithFollowUp} onFollowUpClick={onClick} />);
    fireEvent.click(screen.getByText("Tell me more about Acme"));
    expect(onClick).toHaveBeenCalledWith("Tell me more about Acme");
  });
});

describe("SkipLink", () => {
  it("renders with default text", () => {
    render(<SkipLink />);
    expect(screen.getByText("Skip to main content")).toBeInTheDocument();
  });

  it("renders with custom text", () => {
    render(<SkipLink>Skip to chat</SkipLink>);
    expect(screen.getByText("Skip to chat")).toBeInTheDocument();
  });

  it("links to correct target ID", () => {
    render(<SkipLink targetId="custom-target" />);
    expect(screen.getByRole("link")).toHaveAttribute("href", "#custom-target");
  });

  it("has correct default target ID", () => {
    render(<SkipLink />);
    expect(screen.getByRole("link")).toHaveAttribute("href", "#main-content");
  });
});

describe("Skeleton Components", () => {
  describe("MessageSkeleton", () => {
    it("renders with loading status", () => {
      render(<MessageSkeleton />);
      expect(screen.getByRole("status")).toBeInTheDocument();
    });

    it("has aria-busy attribute", () => {
      render(<MessageSkeleton />);
      expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true");
    });

    it("has descriptive aria-label", () => {
      render(<MessageSkeleton />);
      expect(screen.getByLabelText(/loading message/i)).toBeInTheDocument();
    });
  });

  describe("ChatSkeleton", () => {
    it("renders with loading status", () => {
      render(<ChatSkeleton />);
      expect(screen.getByLabelText(/loading chat/i)).toBeInTheDocument();
    });

    it("has aria-busy attribute", () => {
      render(<ChatSkeleton />);
      expect(screen.getByLabelText(/loading chat/i)).toHaveAttribute("aria-busy", "true");
    });
  });
});
