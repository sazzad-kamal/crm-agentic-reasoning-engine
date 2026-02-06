/**
 * EmailSuggestions - Main component for email follow-up workflow.
 *
 * Three-step flow:
 * 1. Choose category → Show contact list with AI-generated reasons
 * 2. Pick contact → Generate personalized email draft
 * 3. Review & send → Open in email client
 */
import { useEffect } from "react";
import { useEmailSuggestions } from "../hooks/useEmailSuggestions";
import { EmailQuestions } from "./EmailQuestions";
import { EmailContactList } from "./EmailContactList";
import { EmailDraft } from "./EmailDraft";

const STEPS = [
  { id: "questions", label: "Choose category" },
  { id: "contacts", label: "Pick contact" },
  { id: "draft", label: "Review & send" },
];

function StepIndicator({ currentStep }: { currentStep: string }) {
  const currentIndex = STEPS.findIndex((s) => s.id === currentStep);

  return (
    <div className="step-indicator" aria-label="Progress">
      {STEPS.map((step, index) => {
        const isCompleted = index < currentIndex;
        const isCurrent = index === currentIndex;
        return (
          <div
            key={step.id}
            className={`step-indicator__step ${isCompleted ? "step-indicator__step--completed" : ""} ${isCurrent ? "step-indicator__step--current" : ""}`}
          >
            <span className="step-indicator__number">
              {isCompleted ? "✓" : index + 1}
            </span>
            <span className="step-indicator__label">{step.label}</span>
            {index < STEPS.length - 1 && (
              <span className="step-indicator__connector" />
            )}
          </div>
        );
      })}
    </div>
  );
}

export function EmailSuggestions() {
  const {
    view,
    questions,
    selectedCategory,
    contacts,
    generatedEmail,
    loading,
    generating,
    generatingContactId,
    error,
    cachedSecondsAgo,
    refreshing,
    fetchQuestions,
    fetchContacts,
    refreshCache,
    generateEmail,
    regenerateEmail,
    goBack,
    reset,
  } = useEmailSuggestions();

  // Fetch questions on mount
  useEffect(() => {
    fetchQuestions();
  }, [fetchQuestions]);

  const handleQuestionClick = (categoryId: string) => {
    fetchContacts(categoryId);
  };

  const handleContactClick = (contactId: string) => {
    if (selectedCategory) {
      generateEmail(contactId, selectedCategory);
    }
  };

  return (
    <div className="email-suggestions">
      {/* Step indicator - only show after leaving first step */}
      {view !== "questions" && <StepIndicator currentStep={view} />}

      {error && (
        <div className="email-suggestions__error" role="alert">
          <span className="email-suggestions__error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {view === "questions" && (
        <EmailQuestions
          questions={questions}
          loading={loading}
          onQuestionClick={handleQuestionClick}
        />
      )}

      {view === "contacts" && selectedCategory && (
        <EmailContactList
          contacts={contacts}
          category={selectedCategory}
          loading={loading}
          generating={generating}
          selectedContactId={generatingContactId}
          cachedSecondsAgo={cachedSecondsAgo}
          refreshing={refreshing}
          onContactClick={handleContactClick}
          onBack={goBack}
          onRefresh={refreshCache}
        />
      )}

      {view === "draft" && generatedEmail && (
        <EmailDraft
          email={generatedEmail}
          onBack={goBack}
          onReset={reset}
          onRegenerate={!generating ? regenerateEmail : undefined}
        />
      )}
    </div>
  );
}
