/**
 * Hook for fetching data from API endpoints with loading/error states.
 */
import { useState, useEffect } from "react";
import { config } from "../config";
import type { DataResponse } from "../components/dataExplorer/types";

interface UseDataFetchReturn {
  data: DataResponse | null;
  loading: boolean;
  error: string | null;
}

interface FetchState {
  data: DataResponse | null;
  loading: boolean;
  error: string | null;
  endpoint: string;  // Track endpoint to detect changes
}

/**
 * Fetch data from an API endpoint with automatic loading and error handling.
 * 
 * @param endpoint - API endpoint to fetch from (e.g., "/api/data/companies")
 * @returns Object with data, loading state, and error
 */
export function useDataFetch(endpoint: string): UseDataFetchReturn {
  const [state, setState] = useState<FetchState>({
    data: null,
    loading: true,
    error: null,
    endpoint,
  });

  // Derive loading state: true if endpoint changed or still fetching
  const isLoading = state.loading || state.endpoint !== endpoint;

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    fetch(`${config.apiUrl}${endpoint}`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((result) => {
        if (!cancelled) {
          setState({ data: result, loading: false, error: null, endpoint });
        }
      })
      .catch((err) => {
        if (!cancelled && err.name !== "AbortError") {
          setState({ data: null, loading: false, error: err.message, endpoint });
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [endpoint]);

  return { data: state.data, loading: isLoading, error: state.error };
}
