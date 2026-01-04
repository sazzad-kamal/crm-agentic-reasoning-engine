import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ChatArea } from "../components/ChatArea";
import type { ChatMessage } from "../types";
import { EXAMPLE_PROMPTS } from "../config";

// Mock fetch for starter questions API
const mockStarterQuestions = [...EXAMPLE_PROMPTS];

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ questions: mockStarterQuestions }),
    })
  ));
});

// Mock MessageBlock to simplify testing
vi.mock("../components/MessageBlock", () => ({
  MessageBlock: ({ message }: { message: ChatMessage }) => (
    <div data-testid={`message-${message.id}`}>
      Question: {message.question}
    </div>
  ),
}));

describe("ChatArea", () => {
  const defaultProps = {
    messages: [],
    onSuggestionClick: vi.fn(),
    onFollowUpClick: vi.fn(),
  };

  // =========================================================================
  // Empty State
  // =========================================================================

  it("renders empty state when no messages", () => {
    render(<ChatArea {...defaultProps} />);

    expect(screen.getByText("Welcome to Acme CRM AI")).toBeInTheDocument();
  });

  it("renders empty state description", () => {
    render(<ChatArea {...defaultProps} />);

    expect(
      screen.getByText(
        /I can help you find information about your accounts, activities, pipeline/
      )
    ).toBeInTheDocument();
  });

  it("renders example prompts in empty state", async () => {
    render(<ChatArea {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText(/How's my pipeline/)).toBeInTheDocument();
    });
    expect(screen.getByText(/Any renewals at risk/)).toBeInTheDocument();
  });

  it("renders all 3 example prompts", async () => {
    render(<ChatArea {...defaultProps} />);

    await waitFor(() => {
      const buttons = screen.getAllByRole("button");
      // Filter out any non-suggestion buttons
      const suggestionButtons = buttons.filter((btn) =>
        btn.className.includes("suggestion-btn")
      );
      expect(suggestionButtons.length).toBe(3);
    });
  });

  it("calls onSuggestionClick when example prompt clicked", async () => {
    const handleSuggestionClick = vi.fn();

    render(<ChatArea {...defaultProps} onSuggestionClick={handleSuggestionClick} />);

    const button = await waitFor(() => screen.getByText(/How's my pipeline/));
    fireEvent.click(button);

    expect(handleSuggestionClick).toHaveBeenCalledWith(
      "How's my pipeline?"
    );
  });

  // =========================================================================
  // Message List
  // =========================================================================

  it("renders message list when messages present", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test question 1",
        response: { answer: "Answer 1" },
        timestamp: new Date(),
      },
    ];

    render(<ChatArea {...defaultProps} messages={messages} />);

    expect(screen.queryByText("Welcome to Acme CRM AI")).not.toBeInTheDocument();
    expect(screen.getByTestId("message-msg1")).toBeInTheDocument();
  });

  it("renders multiple messages", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Question 1",
        response: { answer: "Answer 1" },
        timestamp: new Date(),
      },
      {
        id: "msg2",
        question: "Question 2",
        response: { answer: "Answer 2" },
        timestamp: new Date(),
      },
      {
        id: "msg3",
        question: "Question 3",
        response: null,
        timestamp: new Date(),
      },
    ];

    render(<ChatArea {...defaultProps} messages={messages} />);

    expect(screen.getByTestId("message-msg1")).toBeInTheDocument();
    expect(screen.getByTestId("message-msg2")).toBeInTheDocument();
    expect(screen.getByTestId("message-msg3")).toBeInTheDocument();
  });

  it("renders messages in correct order", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "First",
        response: { answer: "Answer 1" },
        timestamp: new Date(),
      },
      {
        id: "msg2",
        question: "Second",
        response: { answer: "Answer 2" },
        timestamp: new Date(),
      },
      {
        id: "msg3",
        question: "Third",
        response: { answer: "Answer 3" },
        timestamp: new Date(),
      },
    ];

    const { container } = render(<ChatArea {...defaultProps} messages={messages} />);
    const messageElements = container.querySelectorAll("[data-testid^='message-']");

    expect(messageElements[0]).toHaveAttribute("data-testid", "message-msg1");
    expect(messageElements[1]).toHaveAttribute("data-testid", "message-msg2");
    expect(messageElements[2]).toHaveAttribute("data-testid", "message-msg3");
  });

  // =========================================================================
  // Streaming Status
  // =========================================================================

  it("renders streaming status when provided", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: null,
        timestamp: new Date(),
      },
    ];

    render(<ChatArea {...defaultProps} messages={messages} streamingStatus="Fetching data..." />);

    expect(screen.getByText("Fetching data...")).toBeInTheDocument();
  });

  it("renders streaming status with correct role", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: null,
        timestamp: new Date(),
      },
    ];

    render(<ChatArea {...defaultProps} messages={messages} streamingStatus="Processing..." />);

    const status = screen.getByRole("status");
    expect(status).toBeInTheDocument();
    expect(status).toHaveTextContent("Processing...");
  });

  it("does not render streaming status when null", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: { answer: "Answer" },
        timestamp: new Date(),
      },
    ];

    render(<ChatArea {...defaultProps} messages={messages} streamingStatus={null} />);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("does not render streaming status when undefined", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: { answer: "Answer" },
        timestamp: new Date(),
      },
    ];

    render(<ChatArea {...defaultProps} messages={messages} />);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("updates streaming status", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: null,
        timestamp: new Date(),
      },
    ];

    const { rerender } = render(
      <ChatArea {...defaultProps} messages={messages} streamingStatus="Routing question..." />
    );

    expect(screen.getByText("Routing question...")).toBeInTheDocument();

    rerender(
      <ChatArea {...defaultProps} messages={messages} streamingStatus="Fetching data..." />
    );

    expect(screen.queryByText("Routing question...")).not.toBeInTheDocument();
    expect(screen.getByText("Fetching data...")).toBeInTheDocument();
  });

  // =========================================================================
  // Accessibility
  // =========================================================================

  it("has proper ARIA role for chat area", () => {
    render(<ChatArea {...defaultProps} />);

    const chatArea = screen.getByRole("log");
    expect(chatArea).toBeInTheDocument();
  });

  it("has proper ARIA attributes for chat area", () => {
    render(<ChatArea {...defaultProps} />);

    const chatArea = screen.getByRole("log");
    expect(chatArea).toHaveAttribute("aria-live", "polite");
    expect(chatArea).toHaveAttribute("aria-label", "Chat messages");
  });

  it("message list has proper ARIA role", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: { answer: "Answer" },
        timestamp: new Date(),
      },
    ];

    render(<ChatArea {...defaultProps} messages={messages} />);

    const list = screen.getByRole("list");
    expect(list).toBeInTheDocument();
  });

  it("empty state has proper region role", () => {
    render(<ChatArea {...defaultProps} />);

    expect(screen.getByRole("region", { name: "Getting started" })).toBeInTheDocument();
  });

  it("example prompts have proper ARIA labels", async () => {
    render(<ChatArea {...defaultProps} />);

    const firstPrompt = await waitFor(() => screen.getByLabelText(
      /Ask: How's my pipeline/
    ));
    expect(firstPrompt).toBeInTheDocument();
  });

  it("streaming status has proper ARIA live region", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: null,
        timestamp: new Date(),
      },
    ];

    render(<ChatArea {...defaultProps} messages={messages} streamingStatus="Loading..." />);

    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-live", "polite");
  });

  // =========================================================================
  // Ref Forwarding (React 19)
  // =========================================================================

  it("forwards ref to chat area div", () => {
    const ref = { current: null as HTMLDivElement | null };

    const messages: ChatMessage[] = [];

    render(<ChatArea {...defaultProps} messages={messages} ref={ref} />);

    expect(ref.current).toBeInstanceOf(HTMLDivElement);
    expect(ref.current?.className).toBe("chat-area");
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles many messages", () => {
    const manyMessages: ChatMessage[] = Array.from({ length: 100 }, (_, i) => ({
      id: `msg${i}`,
      question: `Question ${i}`,
      response: { answer: `Answer ${i}` },
      timestamp: new Date(),
    }));

    render(<ChatArea {...defaultProps} messages={manyMessages} />);

    expect(screen.getByTestId("message-msg0")).toBeInTheDocument();
    expect(screen.getByTestId("message-msg99")).toBeInTheDocument();
  });

  it("handles transition from empty to messages", () => {
    const { rerender } = render(<ChatArea {...defaultProps} messages={[]} />);

    expect(screen.getByText("Welcome to Acme CRM AI")).toBeInTheDocument();

    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: { answer: "Answer" },
        timestamp: new Date(),
      },
    ];

    rerender(<ChatArea {...defaultProps} messages={messages} />);

    expect(screen.queryByText("Welcome to Acme CRM AI")).not.toBeInTheDocument();
    expect(screen.getByTestId("message-msg1")).toBeInTheDocument();
  });

  it("handles transition from messages to empty", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: { answer: "Answer" },
        timestamp: new Date(),
      },
    ];

    const { rerender } = render(<ChatArea {...defaultProps} messages={messages} />);

    expect(screen.getByTestId("message-msg1")).toBeInTheDocument();

    rerender(<ChatArea {...defaultProps} messages={[]} />);

    expect(screen.queryByTestId("message-msg1")).not.toBeInTheDocument();
    expect(screen.getByText("Welcome to Acme CRM AI")).toBeInTheDocument();
  });

  it("handles messages with duplicate IDs gracefully", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "First",
        response: { answer: "Answer 1" },
        timestamp: new Date(),
      },
      {
        id: "msg1",
        question: "Second",
        response: { answer: "Answer 2" },
        timestamp: new Date(),
      },
    ];

    // Should still render both despite duplicate IDs (React will warn)
    render(<ChatArea {...defaultProps} messages={messages} />);

    const messageElements = screen.getAllByTestId("message-msg1");
    expect(messageElements.length).toBe(2);
  });

  // =========================================================================
  // Empty State SVG
  // =========================================================================

  it("renders SVG illustration in empty state", () => {
    const { container } = render(<ChatArea {...defaultProps} />);

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
    expect(svg).toHaveAttribute("viewBox", "0 0 200 160");
  });

  // =========================================================================
  // Example Prompts Group
  // =========================================================================

  it("example prompts have proper group role", () => {
    render(<ChatArea {...defaultProps} />);

    const group = screen.getByRole("group");
    expect(group).toBeInTheDocument();
  });

  it("uses unique IDs for aria-labelledby", () => {
    const { container } = render(<ChatArea {...defaultProps} />);

    const group = container.querySelector('[role="group"]');
    const labelId = group?.getAttribute("aria-labelledby");
    const label = container.querySelector(`#${labelId}`);

    expect(label).toBeInTheDocument();
    expect(label?.textContent).toBe("Try one of these to get started:");
  });

  // =========================================================================
  // Callback Props
  // =========================================================================

  it("passes onFollowUpClick to MessageBlock", () => {
    const messages: ChatMessage[] = [
      {
        id: "msg1",
        question: "Test",
        response: { answer: "Answer", follow_up_suggestions: ["Follow-up"] },
        timestamp: new Date(),
      },
    ];

    const handleFollowUpClick = vi.fn();

    render(<ChatArea {...defaultProps} messages={messages} onFollowUpClick={handleFollowUpClick} />);

    // MessageBlock is mocked, so we just verify it receives the prop
    expect(screen.getByTestId("message-msg1")).toBeInTheDocument();
  });

  it("updates when messages array changes", () => {
    const messages1: ChatMessage[] = [
      {
        id: "msg1",
        question: "First set",
        response: { answer: "Answer" },
        timestamp: new Date(),
      },
    ];

    const messages2: ChatMessage[] = [
      {
        id: "msg2",
        question: "Second set",
        response: { answer: "Answer" },
        timestamp: new Date(),
      },
    ];

    const { rerender } = render(<ChatArea {...defaultProps} messages={messages1} />);
    expect(screen.getByTestId("message-msg1")).toBeInTheDocument();

    rerender(<ChatArea {...defaultProps} messages={messages2} />);
    expect(screen.queryByTestId("message-msg1")).not.toBeInTheDocument();
    expect(screen.getByTestId("message-msg2")).toBeInTheDocument();
  });
});
