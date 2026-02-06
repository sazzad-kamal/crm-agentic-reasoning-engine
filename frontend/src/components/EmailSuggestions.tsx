/**
 * EmailSuggestions - Main component for email follow-up workflow.
 *
 * Tabbed interface:
 * - Tabs: Support, Renewals, Billing, Quotes
 * - Contact list shows AI-suggested contacts for selected tab
 * - Draft view for reviewing generated email
 */
import { useEmailSuggestions } from "../hooks/useEmailSuggestions";
import { EmailContactList } from "./EmailContactList";
import { EmailDraft } from "./EmailDraft";

// Tab labels (short for tab buttons)
const TAB_LABELS: Record<string, string> = {
  support: "Support",
  renewals: "Renewals",
  billing: "Billing",
  quotes: "Quotes",
};

// Spinner component
function Spinner() {
  return <span className="email-tab__spinner" aria-hidden="true" />;
}

interface EmailTabsProps {
  categories: readonly string[];
  selectedCategory: string;
  loadedCategories: Set<string>;
  loadingCategory: string | null;
  contactCounts: Record<string, number>;
  onSelect: (category: string) => void;
}

function EmailTabs({
  categories,
  selectedCategory,
  loadedCategories,
  loadingCategory,
  contactCounts,
  onSelect,
}: EmailTabsProps) {
  return (
    <div className="email-tabs" role="tablist" aria-label="Follow-up categories">
      {categories.map((category) => {
        const isLoading = loadingCategory === category;
        const isLoaded = loadedCategories.has(category);
        const isActive = selectedCategory === category;
        const count = contactCounts[category];

        // Tab is disabled only when it's actively loading
        const isDisabled = isLoading;

        // Build class names
        const classNames = [
          "email-tab",
          isActive && "email-tab--active",
          isLoading && "email-tab--loading",
          !isLoaded && !isLoading && "email-tab--unloaded",
        ]
          .filter(Boolean)
          .join(" ");

        return (
          <button
            key={category}
            role="tab"
            aria-selected={isActive}
            aria-controls={`tabpanel-${category}`}
            disabled={isDisabled}
            className={classNames}
            onClick={() => onSelect(category)}
            type="button"
          >
            <span className="email-tab__label">{TAB_LABELS[category] || category}</span>
            {isLoading && <Spinner />}
            {isLoaded && count !== undefined && (
              <span className="email-tab__count">{count}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export function EmailSuggestions() {
  const {
    // Tab state
    categories,
    selectedCategory,
    loadedCategories,
    loadingCategory,

    // Contacts
    contacts,
    contactCounts,

    // Email draft
    view,
    generatedEmail,
    generating,
    generatingContactId,

    // Cache & refresh
    cachedSecondsAgo,
    refreshing,

    // Error
    error,

    // Actions
    selectCategory,
    generateEmail,
    markAsHandled,
    regenerateEmail,
    goBackToList,
    refreshCache,
  } = useEmailSuggestions();

  const handleContactClick = (contactId: string) => {
    generateEmail(contactId);
  };

  const handleOpenInEmail = () => {
    // Mark the current contact as handled when user opens in email client
    if (generatedEmail?.contact.id) {
      markAsHandled(generatedEmail.contact.id);
    }
  };

  const handleReset = () => {
    // Mark as handled and go back to list
    if (generatedEmail?.contact.id) {
      markAsHandled(generatedEmail.contact.id);
    }
    goBackToList();
  };

  // Loading state for initial page load
  const isInitialLoading = loadedCategories.size === 0 && loadingCategory !== null;

  return (
    <div className="email-suggestions">
      {/* Header */}
      <header className="email-suggestions__header">
        <h1 className="email-suggestions__title">Follow-up Inbox</h1>
        <p className="email-suggestions__subtitle">
          AI-suggested contacts who need your attention
        </p>
      </header>

      {/* Error message */}
      {error && (
        <div className="email-suggestions__error" role="alert">
          <span className="email-suggestions__error-icon">!</span>
          <span>{error}</span>
        </div>
      )}

      {/* Tabs - always visible in list view */}
      {view === "list" && (
        <EmailTabs
          categories={categories}
          selectedCategory={selectedCategory}
          loadedCategories={loadedCategories}
          loadingCategory={loadingCategory}
          contactCounts={contactCounts}
          onSelect={selectCategory}
        />
      )}

      {/* Contact list */}
      {view === "list" && (
        <div
          id={`tabpanel-${selectedCategory}`}
          role="tabpanel"
          aria-labelledby={`tab-${selectedCategory}`}
        >
          <EmailContactList
            contacts={contacts}
            category={selectedCategory}
            loading={isInitialLoading || loadingCategory === selectedCategory}
            generating={generating}
            selectedContactId={generatingContactId}
            cachedSecondsAgo={cachedSecondsAgo}
            refreshing={refreshing}
            onContactClick={handleContactClick}
            onRefresh={refreshCache}
          />
        </div>
      )}

      {/* Email draft view */}
      {view === "draft" && generatedEmail && (
        <EmailDraft
          email={generatedEmail}
          onBack={goBackToList}
          onReset={handleReset}
          onRegenerate={!generating ? regenerateEmail : undefined}
          onOpenInEmail={handleOpenInEmail}
        />
      )}
    </div>
  );
}
