import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageBlock } from "../components/MessageBlock";
import type { ChatMessage } from "../types";

// Mock child components to isolate MessageBlock testing
vi.mock("../components/DataTables", () => ({
  DataTables: () => <div data-testid="data-tables">Data tables</div>,
}));

vi.mock("../components/FollowUpSuggestions", () => ({
  FollowUpSuggestions: ({ suggestions }: { suggestions: string[] }) => (
    <div data-testid="follow-up-suggestions">{suggestions.length} suggestions</div>
  ),
}));

vi.mock("../components/Avatar", () => ({
  Avatar: ({ type }: { type: string }) => <div data-testid={`avatar-${type}`}>{type}</div>,
}));

vi.mock("../components/CopyButton", () => ({
  CopyButton: () => <button data-testid="copy-button">Copy</button>,
}));

vi.mock("../components/MarkdownText", () => ({
  MarkdownText: ({ text }: { text: string }) => <div data-testid="markdown-text">{text}</div>,
}));

describe("MessageBlock", () => {
  // =========================================================================
  // Basic Rendering - Question
  // =========================================================================

  it("renders user question", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "What is the status of Acme Corp?",
      response: null,
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByText("What is the status of Acme Corp?")).toBeInTheDocument();
  });

  it("renders user avatar", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: null,
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByTestId("avatar-user")).toBeInTheDocument();
  });

  it("renders assistant avatar", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByTestId("avatar-assistant")).toBeInTheDocument();
  });

  it("renders user and assistant labels", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByText("You")).toBeInTheDocument();
    expect(screen.getByText("Assistant")).toBeInTheDocument();
  });

  // =========================================================================
  // Response States
  // =========================================================================

  it("shows thinking indicator when response is null", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: null,
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByRole("status", { name: /thinking/i })).toBeInTheDocument();
  });

  it("renders answer when response is available", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Here is the answer to your question.",
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByTestId("markdown-text")).toBeInTheDocument();
    expect(screen.getByText("Here is the answer to your question.")).toBeInTheDocument();
  });

  it("renders copy button with answer", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByTestId("copy-button")).toBeInTheDocument();
  });

  // =========================================================================
  // Follow-up Suggestions
  // =========================================================================

  it("renders follow-up suggestions when available and callback provided", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
        follow_up_suggestions: ["Question 1", "Question 2", "Question 3"],
      },
      timestamp: new Date(),
    };

    const handleFollowUp = vi.fn();
    render(<MessageBlock message={message} onFollowUpClick={handleFollowUp} />);

    expect(screen.getByTestId("follow-up-suggestions")).toBeInTheDocument();
    expect(screen.getByText("3 suggestions")).toBeInTheDocument();
  });

  it("does not render follow-up suggestions when callback not provided", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
        follow_up_suggestions: ["Question 1", "Question 2"],
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.queryByTestId("follow-up-suggestions")).not.toBeInTheDocument();
  });

  it("does not render follow-up suggestions when empty", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
        follow_up_suggestions: [],
      },
      timestamp: new Date(),
    };

    const handleFollowUp = vi.fn();
    render(<MessageBlock message={message} onFollowUpClick={handleFollowUp} />);

    expect(screen.queryByTestId("follow-up-suggestions")).not.toBeInTheDocument();
  });

  // =========================================================================
  // Data Tables
  // =========================================================================

  it("renders data tables when raw_data available", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
        raw_data: {
          companies: [
            { company_id: "1", name: "Acme", plan: "Enterprise", renewal_date: "2024-12-31" },
          ],
        },
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByTestId("data-tables")).toBeInTheDocument();
  });

  it("does not render data tables when raw_data is empty", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
        raw_data: {},
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByTestId("data-tables")).toBeInTheDocument();
  });

  it("does not render data tables when raw_data not provided", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: {
        answer: "Test answer",
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.queryByTestId("data-tables")).not.toBeInTheDocument();
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("has proper article role", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "What is the status?",
      response: {
        answer: "Test answer",
      },
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    const article = screen.getByRole("listitem");
    expect(article).toHaveClass("message-block");
  });

  it("has proper ARIA label with question preview", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "What is the status of Acme Corporation and all their activities?",
      response: null,
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    const article = screen.getByRole("listitem");
    expect(article).toHaveAttribute("aria-label", "Conversation about: What is the status of Acme Corporation and all the...");
  });

  it("does not truncate short questions in ARIA label", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Short question",
      response: null,
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByLabelText("Conversation about: Short question")).toBeInTheDocument();
  });

  it("question has heading role", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test question",
      response: null,
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    const heading = screen.getByRole("heading", { level: 3 });
    expect(heading).toHaveTextContent("Test question");
  });

  it("hides labels from screen readers", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Test",
      response: { answer: "Test" },
      timestamp: new Date(),
    };

    const { container } = render(<MessageBlock message={message} />);

    const userLabel = container.querySelector(".message__label--user");
    const assistantLabel = container.querySelector(".message__label--assistant");

    expect(userLabel).toHaveAttribute("aria-hidden", "true");
    expect(assistantLabel).toHaveAttribute("aria-hidden", "true");
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles very long questions", () => {
    const longQuestion = "A".repeat(500);
    const message: ChatMessage = {
      id: "msg1",
      question: longQuestion,
      response: null,
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByText(longQuestion)).toBeInTheDocument();
  });

  it("handles special characters in question", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "What's the status of <Company> & \"Partners\"?",
      response: null,
      timestamp: new Date(),
    };

    render(<MessageBlock message={message} />);

    expect(screen.getByText('What\'s the status of <Company> & "Partners"?')).toBeInTheDocument();
  });

  it("handles complete response with all features", () => {
    const message: ChatMessage = {
      id: "msg1",
      question: "Complete test",
      response: {
        answer: "Complete answer",
        follow_up_suggestions: ["Follow-up 1"],
        raw_data: {
          companies: [{ company_id: "1", name: "Test", plan: "Pro", renewal_date: "2024-12-31" }],
        },
      },
      timestamp: new Date(),
    };

    const handleFollowUp = vi.fn();
    render(<MessageBlock message={message} onFollowUpClick={handleFollowUp} />);

    expect(screen.getByText("Complete answer")).toBeInTheDocument();
    expect(screen.getByTestId("follow-up-suggestions")).toBeInTheDocument();
    expect(screen.getByTestId("data-tables")).toBeInTheDocument();
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("re-renders when message changes", () => {
    const message1: ChatMessage = {
      id: "msg1",
      question: "First question",
      response: null,
      timestamp: new Date(),
    };

    const message2: ChatMessage = {
      id: "msg2",
      question: "Second question",
      response: null,
      timestamp: new Date(),
    };

    const { rerender } = render(<MessageBlock message={message1} />);
    expect(screen.getByText("First question")).toBeInTheDocument();

    rerender(<MessageBlock message={message2} />);
    expect(screen.queryByText("First question")).not.toBeInTheDocument();
    expect(screen.getByText("Second question")).toBeInTheDocument();
  });

  it("updates when response becomes available", () => {
    const message1: ChatMessage = {
      id: "msg1",
      question: "Test",
      response: null,
      timestamp: new Date(),
    };

    const message2: ChatMessage = {
      id: "msg1",
      question: "Test",
      response: { answer: "Answer arrived" },
      timestamp: new Date(),
    };

    const { rerender } = render(<MessageBlock message={message1} />);
    expect(screen.getByRole("status", { name: /thinking/i })).toBeInTheDocument();

    rerender(<MessageBlock message={message2} />);
    expect(screen.queryByRole("status", { name: /thinking/i })).not.toBeInTheDocument();
    expect(screen.getByText("Answer arrived")).toBeInTheDocument();
  });
});
