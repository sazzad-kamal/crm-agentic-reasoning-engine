/**
 * Data Explorer - Browse all CRM data backing the AI.
 * 
 * Main component that provides tabbed navigation, search, and data display
 * for exploring CRM entities (companies, contacts, opportunities, etc.).
 */
import React, { useState, useMemo, useCallback } from "react";
import { useDataFetch } from "../../hooks/useDataFetch";
import { DataTable } from "./DataTable";
import { TABS, type DataTab } from "./types";

// =============================================================================
// Types
// =============================================================================

interface DataExplorerProps {
  onAskAbout?: (question: string) => void;
}

// =============================================================================
// Sub-components
// =============================================================================

function LoadingState() {
  return (
    <div className="data-explorer__loading">
      <div className="data-explorer__spinner" />
      <span>Loading data...</span>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="data-explorer__error">
      <span className="data-explorer__error-icon">⚠️</span>
      <span>Failed to load data: {message}</span>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function DataExplorer({ onAskAbout }: DataExplorerProps) {
  const [activeTab, setActiveTab] = useState<DataTab>("companies");
  const [searchTerm, setSearchTerm] = useState("");

  const currentTab = TABS.find((t) => t.id === activeTab)!;
  const { data, loading, error } = useDataFetch(currentTab.endpoint);

  // Extract data array for cleaner memoization
  const dataArray = data?.data;

  // Filter data based on search
  const filteredData = useMemo(() => {
    if (!dataArray || !searchTerm.trim()) return dataArray || [];

    const term = searchTerm.toLowerCase();
    return dataArray.filter((row) =>
      Object.entries(row).some(([key, val]) => {
        // Don't search nested objects
        if (key.startsWith("_")) return false;
        return String(val).toLowerCase().includes(term);
      })
    );
  }, [dataArray, searchTerm]);

  const handleTabChange = useCallback((tabId: DataTab) => {
    setActiveTab(tabId);
    setSearchTerm("");
  }, []);

  const handleSearch = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  return (
    <div className="data-explorer">
      {/* Header */}
      <div className="data-explorer__header">
        <h2 className="data-explorer__title">
          <span className="data-explorer__icon">📊</span>
          Data Explorer
        </h2>
        <p className="data-explorer__subtitle">
          Browse the CRM data that powers the AI assistant
        </p>
      </div>

      {/* Tabs */}
      <div className="data-explorer__tabs" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`data-tab ${activeTab === tab.id ? "data-tab--active" : ""}`}
            onClick={() => handleTabChange(tab.id)}
          >
            <span className="data-tab__icon">{tab.icon}</span>
            <span className="data-tab__label">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Search */}
      <div className="data-explorer__search">
        <input
          type="text"
          className="data-explorer__search-input"
          placeholder={`Search ${currentTab.label.toLowerCase()}...`}
          value={searchTerm}
          onChange={handleSearch}
          aria-label={`Search ${currentTab.label}`}
        />
        {data && (
          <span className="data-explorer__count">
            {filteredData.length} of {data.total} records
          </span>
        )}
      </div>

      {/* Content */}
      <div className="data-explorer__content" role="tabpanel">
        {loading && <LoadingState />}
        {error && <ErrorState message={error} />}
        {!loading && !error && data && (
          <DataTable
            data={filteredData}
            columns={data.columns}
            onAskAbout={onAskAbout}
            dataType={activeTab}
            nestedFields={currentTab.nestedFields}
          />
        )}
      </div>
    </div>
  );
}

export default DataExplorer;
