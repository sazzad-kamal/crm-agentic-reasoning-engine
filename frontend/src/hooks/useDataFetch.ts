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

/**
 * Fetch data from an API endpoint with automatic loading and error handling.
 * 
 * @param endpoint - API endpoint to fetch from (e.g., "/api/data/companies")
 * @returns Object with data, loading state, and error
 */
export function useDataFetch(endpoint: string): UseDataFetchReturn {
  const [data, setData] = useState<DataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(`${config.apiUrl}${endpoint}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [endpoint]);

  return { data, loading, error };
}
