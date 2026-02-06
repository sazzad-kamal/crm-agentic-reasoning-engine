/**
 * EmailDraft - Premium email preview with proper formatting.
 * Gmail/Outlook-like design with SVG icons.
 */
import type { GeneratedEmail } from "../types";
import { getInitials, getAvatarColor } from "../utils/avatar";

// SVG Icons
const SparkleIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3L14.5 8.5L20 9L16 13.5L17 19L12 16L7 19L8 13.5L4 9L9.5 8.5L12 3Z" />
  </svg>
);

const SendIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const RefreshIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 4 23 10 17 10" />
    <polyline points="1 20 1 14 7 14" />
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
  </svg>
);

const PlusIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

interface EmailDraftProps {
  email: GeneratedEmail;
  onBack: () => void;
  onReset: () => void;
  onRegenerate?: () => void;
  onOpenInEmail?: () => void;
}

export function EmailDraft({ email, onBack, onReset, onRegenerate, onOpenInEmail }: EmailDraftProps) {
  // Convert line breaks to paragraphs for better formatting
  const bodyParagraphs = email.body.split(/\n\n+/).filter(Boolean);

  return (
    <div className="email-draft">
      {/* Header with back button and success badge */}
      <div className="email-draft__header">
        <button
          type="button"
          className="email-draft__back"
          onClick={onBack}
          aria-label="Go back to contact list"
        >
          ← Back
        </button>
        <div className="email-draft__badge">
          <span className="email-draft__badge-icon"><SparkleIcon /></span>
          <span className="email-draft__badge-text">Draft Ready</span>
        </div>
      </div>

      {/* Email preview card */}
      <div className="email-draft__card">
        {/* Recipient section */}
        <div className="email-draft__recipient">
          <div
            className="email-draft__avatar"
            style={{ backgroundColor: getAvatarColor(email.contact.name) }}
          >
            {getInitials(email.contact.name)}
          </div>
          <div className="email-draft__recipient-info">
            <div className="email-draft__recipient-name">{email.contact.name}</div>
            <div className="email-draft__recipient-email">{email.contact.email}</div>
            {email.contact.company && (
              <div className="email-draft__recipient-company">{email.contact.company}</div>
            )}
          </div>
        </div>

        {/* Subject line */}
        <div className="email-draft__subject">
          <span className="email-draft__subject-label">Subject</span>
          <span className="email-draft__subject-text">{email.subject}</span>
        </div>

        {/* Email body with proper paragraph formatting */}
        <div className="email-draft__body">
          {bodyParagraphs.map((paragraph, index) => (
            <p key={index} className="email-draft__paragraph">
              {paragraph.split('\n').map((line, lineIndex) => (
                <span key={lineIndex}>
                  {line}
                  {lineIndex < paragraph.split('\n').length - 1 && <br />}
                </span>
              ))}
            </p>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="email-draft__actions">
        <a
          href={email.mailtoLink}
          className="email-draft__send-btn"
          target="_blank"
          rel="noopener noreferrer"
          onClick={onOpenInEmail}
        >
          <span className="email-draft__send-icon"><SendIcon /></span>
          <span className="email-draft__send-text">Open in Email Client</span>
          <span className="email-draft__send-arrow">→</span>
        </a>
        <div className="email-draft__secondary-actions">
          {onRegenerate && (
            <button
              type="button"
              className="email-draft__action-btn"
              onClick={onRegenerate}
            >
              <RefreshIcon /> Regenerate
            </button>
          )}
          <button
            type="button"
            className="email-draft__action-btn"
            onClick={onReset}
          >
            <PlusIcon /> Draft Another
          </button>
        </div>
      </div>

      <p className="email-draft__hint">
        Your email client will open with this draft ready to review and send.
      </p>
    </div>
  );
}
