/**
 * Hook for email suggestion workflow with tabbed interface.
 *
 * Features:
 * - Tab-based category navigation (Support, Renewals, Billing, Quotes)
 * - Per-category contact caching
 * - Background prefetch of remaining tabs after first loads
 * - Cross-tab deduplication via handled contact tracking
 */
import { useState, useCallback, useEffect, useRef } from "react";
import { endpoints } from "../config";
import type { EmailContact, GeneratedEmail, EmailContactsResponse } from "../types";

// Fixed category order
const CATEGORIES = ["support", "renewals", "billing", "quotes"] as const;
type Category = (typeof CATEGORIES)[number];

type EmailView = "list" | "draft";

interface UseEmailSuggestionsReturn {
  // Tab state
  categories: readonly string[];
  selectedCategory: string;
  loadedCategories: Set<string>;
  loadingCategory: string | null;

  // Contacts (filtered by handled)
  contacts: EmailContact[];
  contactCounts: Record<string, number>;

  // Handled tracking
  handledContactIds: Set<string>;

  // Email draft
  view: EmailView;
  generatedEmail: GeneratedEmail | null;
  generating: boolean;
  generatingContactId: string | null;

  // Cache & refresh
  cachedSecondsAgo: number | null;
  refreshing: boolean;

  // Error
  error: string | null;

  // Actions
  selectCategory: (category: string) => void;
  generateEmail: (contactId: string) => Promise<void>;
  markAsHandled: (contactId: string) => void;
  regenerateEmail: () => Promise<void>;
  goBackToList: () => void;
  refreshCache: () => Promise<void>;
}

export function useEmailSuggestions(): UseEmailSuggestionsReturn {
  // Tab state
  const [selectedCategory, setSelectedCategory] = useState<Category>(CATEGORIES[0]);
  const [loadedCategories, setLoadedCategories] = useState<Set<string>>(new Set());
  const [loadingCategory, setLoadingCategory] = useState<string | null>(null);

  // Per-category contact storage
  const [contactsByCategory, setContactsByCategory] = useState<Record<string, EmailContact[]>>({});

  // Handled tracking (session-based, cross-tab dedup)
  const [handledContactIds, setHandledContactIds] = useState<Set<string>>(new Set());

  // Email draft state
  const [view, setView] = useState<EmailView>("list");
  const [generatedEmail, setGeneratedEmail] = useState<GeneratedEmail | null>(null);
  const [generating, setGenerating] = useState(false);
  const [generatingContactId, setGeneratingContactId] = useState<string | null>(null);
  const [selectedContactId, setSelectedContactId] = useState<string | null>(null);

  // Cache & refresh
  const [cachedSecondsAgo, setCachedSecondsAgo] = useState<number | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Error
  const [error, setError] = useState<string | null>(null);

  // Track if prefetch is in progress to avoid interrupting user clicks
  const prefetchInProgressRef = useRef(false);

  // Fetch contacts for a specific category
  const fetchContacts = useCallback(async (category: string, options?: { background?: boolean }) => {
    const isBackground = options?.background ?? false;

    // Don't set loading state for background prefetch
    if (!isBackground) {
      setLoadingCategory(category);
    }
    setError(null);

    try {
      const res = await fetch(
        `${endpoints.emailContacts}?category=${encodeURIComponent(category)}`
      );
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const data: EmailContactsResponse = await res.json();

      setContactsByCategory((prev) => ({
        ...prev,
        [category]: data.contacts,
      }));
      setLoadedCategories((prev) => new Set([...prev, category]));

      // Only update cache age for the currently selected category
      if (category === selectedCategory || !isBackground) {
        setCachedSecondsAgo(data.cachedSecondsAgo);
      }
    } catch (err) {
      if (!isBackground) {
        setError(err instanceof Error ? err.message : "Failed to fetch contacts");
      }
      // For background fetches, silently fail (user can click tab to retry)
    } finally {
      if (!isBackground) {
        setLoadingCategory(null);
      }
    }
  }, [selectedCategory]);

  // Background prefetch of remaining categories
  const prefetchRemainingCategories = useCallback(async () => {
    if (prefetchInProgressRef.current) return;
    prefetchInProgressRef.current = true;

    const remaining = CATEGORIES.filter((c) => !loadedCategories.has(c));

    for (const category of remaining) {
      // Stop prefetch if user is actively loading a tab
      if (loadingCategory) break;

      await fetchContacts(category, { background: true });

      // Small delay between background fetches to avoid overwhelming the server
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    prefetchInProgressRef.current = false;
  }, [loadedCategories, loadingCategory, fetchContacts]);

  // Auto-load first category on mount
  useEffect(() => {
    if (loadedCategories.size === 0 && !loadingCategory) {
      fetchContacts(CATEGORIES[0]);
    }
  }, [loadedCategories.size, loadingCategory, fetchContacts]);

  // Start background prefetch after first category loads
  useEffect(() => {
    if (loadedCategories.size === 1 && !loadingCategory && !prefetchInProgressRef.current) {
      prefetchRemainingCategories();
    }
  }, [loadedCategories.size, loadingCategory, prefetchRemainingCategories]);

  // Select a category (tab click)
  const selectCategory = useCallback((category: string) => {
    if (!CATEGORIES.includes(category as Category)) return;
    if (loadingCategory === category) return; // Already loading this one

    setSelectedCategory(category as Category);
    setError(null);

    // If not loaded yet, fetch it
    if (!loadedCategories.has(category)) {
      fetchContacts(category);
    } else {
      // Update cache age from stored data (approximate)
      setCachedSecondsAgo(cachedSecondsAgo);
    }
  }, [loadedCategories, loadingCategory, fetchContacts, cachedSecondsAgo]);

  // Generate email for a contact
  const generateEmail = useCallback(async (contactId: string) => {
    setGenerating(true);
    setGeneratingContactId(contactId);
    setSelectedContactId(contactId);
    setError(null);

    try {
      const res = await fetch(endpoints.emailGenerate, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contactId, category: selectedCategory }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const data: GeneratedEmail = await res.json();
      setGeneratedEmail(data);
      setView("draft");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate email");
    } finally {
      setGenerating(false);
      setGeneratingContactId(null);
    }
  }, [selectedCategory]);

  // Mark a contact as handled (called when user clicks "Open in Email")
  const markAsHandled = useCallback((contactId: string) => {
    setHandledContactIds((prev) => new Set([...prev, contactId]));
  }, []);

  // Regenerate email for current contact
  const regenerateEmail = useCallback(async () => {
    if (!selectedContactId) return;

    setGenerating(true);
    setError(null);

    try {
      const res = await fetch(endpoints.emailGenerate, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contactId: selectedContactId, category: selectedCategory }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const data: GeneratedEmail = await res.json();
      setGeneratedEmail(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to regenerate email");
    } finally {
      setGenerating(false);
    }
  }, [selectedContactId, selectedCategory]);

  // Go back to list view
  const goBackToList = useCallback(() => {
    setGeneratedEmail(null);
    setSelectedContactId(null);
    setView("list");
    setError(null);
  }, []);

  // Refresh cache (clears all categories, re-fetches current)
  const refreshCache = useCallback(async () => {
    setRefreshing(true);
    setError(null);

    try {
      // Clear cache on backend
      await fetch(endpoints.emailRefresh, { method: "POST" });

      // Clear local cache
      setContactsByCategory({});
      setLoadedCategories(new Set());

      // Re-fetch current category
      const res = await fetch(
        `${endpoints.emailContacts}?category=${encodeURIComponent(selectedCategory)}`
      );
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const data: EmailContactsResponse = await res.json();

      setContactsByCategory({ [selectedCategory]: data.contacts });
      setLoadedCategories(new Set([selectedCategory]));
      setCachedSecondsAgo(data.cachedSecondsAgo);

      // Restart background prefetch
      prefetchInProgressRef.current = false;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh data");
    } finally {
      setRefreshing(false);
    }
  }, [selectedCategory]);

  // Compute visible contacts (filtered by handled)
  const currentCategoryContacts = contactsByCategory[selectedCategory] || [];
  const visibleContacts = currentCategoryContacts.filter(
    (c) => !handledContactIds.has(c.contactId)
  );

  // Compute counts for all loaded categories (filtered by handled)
  const contactCounts: Record<string, number> = {};
  for (const category of CATEGORIES) {
    if (loadedCategories.has(category)) {
      const categoryContacts = contactsByCategory[category] || [];
      contactCounts[category] = categoryContacts.filter(
        (c) => !handledContactIds.has(c.contactId)
      ).length;
    }
  }

  return {
    // Tab state
    categories: CATEGORIES,
    selectedCategory,
    loadedCategories,
    loadingCategory,

    // Contacts
    contacts: visibleContacts,
    contactCounts,

    // Handled tracking
    handledContactIds,

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
  };
}
