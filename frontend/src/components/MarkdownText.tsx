import { memo, useMemo, ComponentPropsWithoutRef } from "react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownTextProps {
  text: string;
  className?: string;
}

/**
 * Custom component overrides for react-markdown
 * Maps markdown elements to styled HTML with proper ARIA and semantic markup
 */
const markdownComponents: Components = {
  // Headings with proper hierarchy
  h1: ({ children, ...props }) => (
    <h2 className="md-h2" {...props}>{children}</h2>
  ),
  h2: ({ children, ...props }) => (
    <h2 className="md-h2" {...props}>{children}</h2>
  ),
  h3: ({ children, ...props }) => (
    <h3 className="md-h3" {...props}>{children}</h3>
  ),
  h4: ({ children, ...props }) => (
    <h4 className="md-h4" {...props}>{children}</h4>
  ),

  // Paragraphs
  p: ({ children, ...props }) => (
    <p className="md-p" {...props}>{children}</p>
  ),

  // Inline code
  code: ({ children, className, ...props }: ComponentPropsWithoutRef<"code"> & { className?: string }) => {
    // Check if this is a code block (has language class) or inline code
    const isCodeBlock = className?.includes("language-");

    if (isCodeBlock) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }

    return (
      <code className="md-code" {...props}>
        {children}
      </code>
    );
  },

  // Code blocks
  pre: ({ children, ...props }) => (
    <pre className="md-pre" {...props}>
      {children}
    </pre>
  ),

  // Lists
  ul: ({ children, ...props }) => (
    <ul className="md-ul" role="list" {...props}>{children}</ul>
  ),
  ol: ({ children, ...props }) => (
    <ol className="md-ol" role="list" {...props}>{children}</ol>
  ),
  li: ({ children, ...props }) => (
    <li className="md-li" {...props}>{children}</li>
  ),

  // Links - open in new tab with security attributes
  a: ({ children, href, ...props }) => (
    <a
      href={href}
      className="md-link"
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    >
      {children}
    </a>
  ),

  // Blockquotes
  blockquote: ({ children, ...props }) => (
    <blockquote className="md-blockquote" {...props}>{children}</blockquote>
  ),

  // Tables with proper accessibility
  table: ({ children, ...props }) => (
    <div className="md-table-wrapper" role="region" aria-label="Data table">
      <table className="md-table" {...props}>{children}</table>
    </div>
  ),
  thead: ({ children, ...props }) => (
    <thead className="md-thead" {...props}>{children}</thead>
  ),
  tbody: ({ children, ...props }) => (
    <tbody className="md-tbody" {...props}>{children}</tbody>
  ),
  tr: ({ children, ...props }) => (
    <tr className="md-tr" {...props}>{children}</tr>
  ),
  th: ({ children, ...props }) => (
    <th className="md-th" scope="col" {...props}>{children}</th>
  ),
  td: ({ children, ...props }) => (
    <td className="md-td" {...props}>{children}</td>
  ),

  // Horizontal rule
  hr: (props) => (
    <hr className="md-hr" {...props} />
  ),

  // Strong and emphasis
  strong: ({ children, ...props }) => (
    <strong className="md-strong" {...props}>{children}</strong>
  ),
  em: ({ children, ...props }) => (
    <em className="md-em" {...props}>{children}</em>
  ),
};

/**
 * Secure markdown renderer for chat messages
 * Uses react-markdown to prevent XSS vulnerabilities
 * Supports GFM (GitHub Flavored Markdown): tables, strikethrough, autolinks, task lists
 */
export const MarkdownText = memo(function MarkdownText({
  text,
  className = ""
}: MarkdownTextProps) {
  // Memoize the remark plugins array to prevent re-renders
  const remarkPlugins = useMemo(() => [remarkGfm], []);

  return (
    <div className={`markdown-text ${className}`}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        components={markdownComponents}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
