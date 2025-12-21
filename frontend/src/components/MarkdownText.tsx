import { memo, useMemo } from "react";

interface MarkdownTextProps {
  text: string;
  className?: string;
}

/**
 * Simple markdown renderer for chat messages
 * Supports: **bold**, *italic*, `code`, - lists, numbered lists, ### headers
 */
export const MarkdownText = memo(function MarkdownText({ text, className = "" }: MarkdownTextProps) {
  const rendered = useMemo(() => {
    return parseMarkdown(text);
  }, [text]);

  return (
    <div 
      className={`markdown-text ${className}`}
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  );
});

/**
 * Parse markdown text to HTML
 */
function parseMarkdown(text: string): string {
  if (!text) return "";

  let html = escapeHtml(text);

  // Headers (### Header)
  html = html.replace(/^### (.+)$/gm, '<h4 class="md-h4">$1</h4>');
  html = html.replace(/^## (.+)$/gm, '<h3 class="md-h3">$1</h3>');
  html = html.replace(/^# (.+)$/gm, '<h2 class="md-h2">$1</h2>');

  // Bold (**text** or __text__)
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

  // Italic (*text* or _text_)
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
  html = html.replace(/(?<!_)_(?!_)(.+?)(?<!_)_(?!_)/g, '<em>$1</em>');

  // Inline code (`code`)
  html = html.replace(/`([^`]+)`/g, '<code class="md-code">$1</code>');

  // Code blocks (```code```)
  html = html.replace(/```([\s\S]*?)```/g, '<pre class="md-pre"><code>$1</code></pre>');

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li class="md-li">$1</li>');
  html = html.replace(/(<li class="md-li">.*<\/li>\n?)+/g, '<ul class="md-ul">$&</ul>');

  // Ordered lists (1. item)
  html = html.replace(/^\d+\. (.+)$/gm, '<li class="md-li-num">$1</li>');
  html = html.replace(/(<li class="md-li-num">.*<\/li>\n?)+/g, '<ol class="md-ol">$&</ol>');

  // Line breaks (preserve double newlines as paragraphs)
  html = html.replace(/\n\n/g, '</p><p class="md-p">');
  html = html.replace(/\n/g, '<br />');

  // Wrap in paragraph if not already structured
  if (!html.startsWith('<')) {
    html = `<p class="md-p">${html}</p>`;
  }

  return html;
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
}
