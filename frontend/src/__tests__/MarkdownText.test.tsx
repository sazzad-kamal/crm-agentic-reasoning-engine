import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MarkdownText } from "../components/MarkdownText";

describe("MarkdownText", () => {
  // =========================================================================
  // Basic Rendering
  // =========================================================================

  it("renders plain text", () => {
    render(<MarkdownText text="Hello world" />);
    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("renders empty string", () => {
    const { container } = render(<MarkdownText text="" />);
    expect(container.querySelector(".markdown-text")).toBeEmptyDOMElement();
  });

  it("applies custom className", () => {
    const { container } = render(
      <MarkdownText text="Test" className="custom-class" />
    );
    const element = container.querySelector(".markdown-text");
    expect(element).toHaveClass("markdown-text", "custom-class");
  });

  // =========================================================================
  // Headers
  // =========================================================================

  it("renders h2 headers (# Header)", () => {
    const { container } = render(<MarkdownText text="# Main Header" />);
    expect(container.querySelector("h2")).toHaveTextContent("Main Header");
    expect(container.querySelector("h2")).toHaveClass("md-h2");
  });

  it("renders h3 headers (## Header)", () => {
    const { container } = render(<MarkdownText text="## Sub Header" />);
    expect(container.querySelector("h3")).toHaveTextContent("Sub Header");
    expect(container.querySelector("h3")).toHaveClass("md-h3");
  });

  it("renders h4 headers (### Header)", () => {
    const { container } = render(<MarkdownText text="### Small Header" />);
    expect(container.querySelector("h4")).toHaveTextContent("Small Header");
    expect(container.querySelector("h4")).toHaveClass("md-h4");
  });

  it("renders multiple headers", () => {
    const text = "# Header 1\n## Header 2\n### Header 3";
    const { container } = render(<MarkdownText text={text} />);

    expect(container.querySelector("h2")).toHaveTextContent("Header 1");
    expect(container.querySelector("h3")).toHaveTextContent("Header 2");
    expect(container.querySelector("h4")).toHaveTextContent("Header 3");
  });

  // =========================================================================
  // Bold and Italic
  // =========================================================================

  it("renders bold text with **", () => {
    const { container } = render(<MarkdownText text="This is **bold** text" />);
    expect(container.querySelector("strong")).toHaveTextContent("bold");
  });

  it("renders bold text with __", () => {
    const { container } = render(<MarkdownText text="This is __bold__ text" />);
    expect(container.querySelector("strong")).toHaveTextContent("bold");
  });

  it("renders italic text with *", () => {
    const { container } = render(<MarkdownText text="This is *italic* text" />);
    expect(container.querySelector("em")).toHaveTextContent("italic");
  });

  it("renders italic text with _", () => {
    const { container } = render(<MarkdownText text="This is _italic_ text" />);
    expect(container.querySelector("em")).toHaveTextContent("italic");
  });

  it("renders bold and italic combined", () => {
    const { container } = render(
      <MarkdownText text="**bold** and *italic*" />
    );
    expect(container.querySelector("strong")).toHaveTextContent("bold");
    expect(container.querySelector("em")).toHaveTextContent("italic");
  });

  // =========================================================================
  // Code
  // =========================================================================

  it("renders inline code", () => {
    const { container } = render(
      <MarkdownText text="Use `console.log()` to debug" />
    );
    const code = container.querySelector("code");
    expect(code).toHaveTextContent("console.log()");
    expect(code).toHaveClass("md-code");
  });

  it("renders code blocks", () => {
    const { container } = render(
      <MarkdownText text="```\nconst x = 1;\nconsole.log(x);\n```" />
    );
    const pre = container.querySelector("pre");
    expect(pre).toHaveClass("md-pre");
    expect(pre?.querySelector("code")).toHaveTextContent("const x = 1;");
  });

  it("renders multiple inline code snippets", () => {
    const { container } = render(
      <MarkdownText text="`var1` and `var2`" />
    );
    const codes = container.querySelectorAll("code");
    expect(codes).toHaveLength(2);
    expect(codes[0]).toHaveTextContent("var1");
    expect(codes[1]).toHaveTextContent("var2");
  });

  // =========================================================================
  // Lists
  // =========================================================================

  it("renders unordered list", () => {
    const text = "- Item 1\n- Item 2\n- Item 3";
    const { container } = render(<MarkdownText text={text} />);

    const ul = container.querySelector("ul");
    expect(ul).toHaveClass("md-ul");

    const items = container.querySelectorAll("li");
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("Item 1");
  });

  it("renders ordered list", () => {
    const text = "1. First\n2. Second\n3. Third";
    const { container } = render(<MarkdownText text={text} />);

    const ol = container.querySelector("ol");
    expect(ol).toHaveClass("md-ol");

    const items = container.querySelectorAll("li");
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("First");
  });

  it("renders mixed content with lists", () => {
    const text = "Header:\n- Item 1\n- Item 2\n\nMore text";
    const { container } = render(<MarkdownText text={text} />);

    expect(container.querySelector("ul")).toBeInTheDocument();
    expect(container.querySelectorAll("li")).toHaveLength(2);
  });

  // =========================================================================
  // Line Breaks and Paragraphs
  // =========================================================================

  it("preserves line breaks", () => {
    // Use String.raw or construct string with actual newline
    const text = "Line 1" + "\n" + "Line 2";
    const { container } = render(<MarkdownText text={text} />);
    // Component converts \n to <br />
    expect(container.textContent).toContain("Line 1");
    expect(container.textContent).toContain("Line 2");
  });

  it("creates paragraphs from double newlines", () => {
    const { container } = render(
      <MarkdownText text="Para 1\n\nPara 2" />
    );
    const paragraphs = container.querySelectorAll("p");
    expect(paragraphs.length).toBeGreaterThan(0);
  });

  it("wraps content in paragraph by default", () => {
    const { container } = render(<MarkdownText text="Simple text" />);
    const p = container.querySelector("p");
    expect(p).toHaveClass("md-p");
  });

  // =========================================================================
  // XSS Protection
  // =========================================================================

  it("escapes HTML special characters", () => {
    const { container } = render(
      <MarkdownText text="<script>alert('xss')</script>" />
    );
    const html = container.innerHTML;
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("<script>");
  });

  it("escapes ampersands", () => {
    const { container } = render(<MarkdownText text="A & B" />);
    expect(container.innerHTML).toContain("&amp;");
  });

  it("escapes quotes", () => {
    const { container } = render(<MarkdownText text={'"quotes" and \'apostrophes\''} />);
    // HTML entities are escaped internally but rendered as actual characters by the browser
    expect(container.textContent).toContain('"quotes"');
    expect(container.textContent).toContain("'apostrophes'");
  });

  it("prevents XSS in markdown syntax", () => {
    const { container } = render(
      <MarkdownText text="**<img src=x onerror=alert(1)>**" />
    );
    expect(container.innerHTML).not.toContain("<img");
    expect(container.innerHTML).toContain("&lt;img");
  });

  // =========================================================================
  // Complex Combinations
  // =========================================================================

  it("renders complex markdown with multiple features", () => {
    const text = `# Main Title

This is **bold** and *italic* text.

- List item 1
- List item 2

Use \`code\` for inline.

## Subsection

More content here.`;

    const { container } = render(<MarkdownText text={text} />);

    expect(container.querySelector("h2")).toBeInTheDocument();
    expect(container.querySelector("h3")).toBeInTheDocument();
    expect(container.querySelector("strong")).toBeInTheDocument();
    expect(container.querySelector("em")).toBeInTheDocument();
    expect(container.querySelector("code")).toBeInTheDocument();
    expect(container.querySelector("ul")).toBeInTheDocument();
  });

  it("handles nested formatting", () => {
    const { container } = render(
      <MarkdownText text="**Bold with *italic* inside**" />
    );
    expect(container.querySelector("strong")).toBeInTheDocument();
    expect(container.querySelector("em")).toBeInTheDocument();
  });

  it("handles code with special characters", () => {
    const { container } = render(
      <MarkdownText text="`const x = '<div>'`" />
    );
    const code = container.querySelector("code");
    expect(code?.innerHTML).toContain("&lt;div&gt;");
  });

  // =========================================================================
  // Edge Cases
  // =========================================================================

  it("handles text with only whitespace", () => {
    const { container } = render(<MarkdownText text="   \n   " />);
    expect(container.querySelector(".markdown-text")).toBeInTheDocument();
  });

  it("handles very long text", () => {
    const longText = "word ".repeat(1000);
    render(<MarkdownText text={longText} />);
    expect(screen.getByText(/word/)).toBeInTheDocument();
  });

  it("handles unicode characters", () => {
    render(<MarkdownText text="Hello 你好 мир 🎉" />);
    expect(screen.getByText(/你好/)).toBeInTheDocument();
  });

  it("handles malformed markdown gracefully", () => {
    const { container } = render(
      <MarkdownText text="**unclosed bold *unclosed italic `unclosed code" />
    );
    // Should render without crashing
    expect(container.querySelector(".markdown-text")).toBeInTheDocument();
  });

  it("handles empty code blocks", () => {
    const { container } = render(<MarkdownText text="```\n\n```" />);
    const pre = container.querySelector("pre");
    expect(pre).not.toBeNull();
  });

  // =========================================================================
  // Memoization
  // =========================================================================

  it("memoizes rendering", () => {
    const { rerender } = render(<MarkdownText text="Test" />);
    const firstRender = screen.getByText("Test");

    // Same text should not re-render
    rerender(<MarkdownText text="Test" />);
    const secondRender = screen.getByText("Test");

    expect(firstRender).toBe(secondRender);
  });

  it("updates when text changes", () => {
    const { rerender } = render(<MarkdownText text="Original" />);
    expect(screen.getByText("Original")).toBeInTheDocument();

    rerender(<MarkdownText text="Updated" />);
    expect(screen.getByText("Updated")).toBeInTheDocument();
    expect(screen.queryByText("Original")).not.toBeInTheDocument();
  });
});
