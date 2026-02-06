/**
 * EmailContactList - List of contacts with avatars and AI-generated reasons.
 * Displays contacts for the selected tab with loading skeletons.
 */
import type { KeyboardEvent } from "react";
import type { EmailContact } from "../types";
import { getInitials, getAvatarColor } from "../utils/avatar";

interface EmailContactListProps {
  contacts: EmailContact[];
  category: string;
  loading: boolean;
  generating: boolean;
  selectedContactId: string | null;
  cachedSecondsAgo: number | null;
  refreshing: boolean;
  onContactClick: (contactId: string) => void;
  onRefresh: () => void;
}

function formatCacheAge(seconds: number): string {
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes === 1) return "1 min ago";
  return `${minutes} min ago`;
}

const CATEGORY_CONFIG: Record<string, { title: string; description: string }> = {
  support: {
    title: "Support Follow-up",
    description: "Customers who recently had support interactions",
  },
  renewals: {
    title: "Upcoming Renewals",
    description: "Contacts with renewals coming up soon",
  },
  billing: {
    title: "Billing Inquiries",
    description: "Contacts with billing questions or payment follow-ups",
  },
  quotes: {
    title: "Open Quotes",
    description: "Contacts with pending quotes that need follow-up",
  },
};

function LoadingSkeleton() {
  return (
    <div className="email-contact-list email-contact-list--loading">
      <div className="email-contact-list__loading-state">
        <div className="email-contact-list__loading-spinner" />
        <p className="email-contact-list__loading-text">
          Finding contacts who need follow-up...
        </p>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="email-contact-list__empty">
      <div className="email-contact-list__empty-illustration">
        <svg viewBox="0 0 200 160" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Checkmark circle */}
          <circle cx="100" cy="70" r="40" fill="#F0FDF4" stroke="#86EFAC" strokeWidth="3"/>
          <path d="M80 70L93 83L120 56" stroke="#22C55E" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/>
          {/* Decorative elements */}
          <circle cx="155" cy="45" r="10" fill="#EEF2FF" stroke="#C7D2FE" strokeWidth="2"/>
          <circle cx="50" cy="50" r="8" fill="#FEF3C7" stroke="#FCD34D" strokeWidth="2"/>
          {/* Sparkles */}
          <path d="M165 75L166.5 79L170.5 80.5L166.5 82L165 86L163.5 82L159.5 80.5L163.5 79L165 75Z" fill="#A5B4FC"/>
          <path d="M40 80L41.5 84L45.5 85.5L41.5 87L40 91L38.5 87L34.5 85.5L38.5 84L40 80Z" fill="#FCD34D"/>
        </svg>
      </div>
      <h3 className="email-contact-list__empty-title">All caught up!</h3>
      <p className="email-contact-list__empty-text">
        No follow-ups needed in this category right now.
      </p>
    </div>
  );
}

export function EmailContactList({
  contacts,
  category,
  loading,
  generating,
  selectedContactId,
  cachedSecondsAgo,
  refreshing,
  onContactClick,
  onRefresh,
}: EmailContactListProps) {
  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, contactId: string) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onContactClick(contactId);
    }
  };

  const config = CATEGORY_CONFIG[category] || {
    title: category,
    description: "Contacts who need follow-up",
  };

  if (loading) {
    return <LoadingSkeleton />;
  }

  return (
    <div className="email-contact-list">
      {/* Header with title and cache info */}
      <div className="email-contact-list__header">
        <div className="email-contact-list__header-text">
          <h2 className="email-contact-list__title">{config.title}</h2>
          <p className="email-contact-list__description">{config.description}</p>
        </div>
        {cachedSecondsAgo !== null && (
          <div className="email-contact-list__cache-info">
            <span className="email-contact-list__cache-age">
              Data from {formatCacheAge(cachedSecondsAgo)}
            </span>
            <button
              type="button"
              className="email-contact-list__refresh"
              onClick={onRefresh}
              disabled={refreshing}
              aria-label="Refresh data"
            >
              {refreshing ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        )}
      </div>

      {contacts.length === 0 ? (
        <EmptyState />
      ) : (
        <>
          <p className="email-contact-list__hint">
            Click a contact to generate a personalized email draft
          </p>
          <div className="email-contact-list__items" role="list">
            {contacts.map((contact) => {
              const isSelected = generating && selectedContactId === contact.contactId;
              return (
                <button
                  key={contact.contactId}
                  className={`email-contact-card ${isSelected ? "email-contact-card--generating" : ""}`}
                  onClick={() => onContactClick(contact.contactId)}
                  onKeyDown={(e) => handleKeyDown(e, contact.contactId)}
                  type="button"
                  disabled={generating}
                  role="listitem"
                >
                  <div
                    className="email-contact-card__avatar"
                    style={{ backgroundColor: getAvatarColor(contact.name) }}
                  >
                    {getInitials(contact.name)}
                  </div>
                  <div className="email-contact-card__content">
                    <div className="email-contact-card__name">{contact.name}</div>
                    {contact.company && (
                      <div className="email-contact-card__company">{contact.company}</div>
                    )}
                    <div className="email-contact-card__reason">{contact.reason}</div>
                  </div>
                  <div className="email-contact-card__meta">
                    <span className="email-contact-card__time">
                      {contact.lastContactAgo || contact.lastContact || ""}
                    </span>
                    {isSelected ? (
                      <span className="email-contact-card__status">
                        <span className="email-contact-card__spinner" />
                        Writing...
                      </span>
                    ) : (
                      <span className="email-contact-card__cta">Draft email →</span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
