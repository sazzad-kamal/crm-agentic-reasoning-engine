import { describe, it, expect } from "vitest";
import { config, endpoints, EXAMPLE_PROMPTS } from "../config";

describe("config", () => {
  // =========================================================================
  // API URL Configuration
  // =========================================================================

  it("has apiUrl defined", () => {
    expect(config.apiUrl).toBeDefined();
    expect(typeof config.apiUrl).toBe("string");
    expect(config.apiUrl.length).toBeGreaterThan(0);
  });

  it("apiUrl is a valid URL format", () => {
    expect(config.apiUrl).toMatch(/^https?:\/\/.+/);
  });

  // =========================================================================
  // Feature Flags
  // =========================================================================

  it("has all feature flags defined", () => {
    expect(config.features).toBeDefined();
    expect(config.features.showDataTables).toBeDefined();
    expect(config.features.showSteps).toBeDefined();
    expect(config.features.showFollowUpSuggestions).toBeDefined();
    expect(config.features.showSources).toBeDefined();
    expect(config.features.showLatency).toBeDefined();
  });

  it("all feature flags are booleans", () => {
    expect(typeof config.features.showDataTables).toBe("boolean");
    expect(typeof config.features.showSteps).toBe("boolean");
    expect(typeof config.features.showFollowUpSuggestions).toBe("boolean");
    expect(typeof config.features.showSources).toBe("boolean");
    expect(typeof config.features.showLatency).toBe("boolean");
  });

  it("has expected feature flags enabled by default", () => {
    expect(config.features.showDataTables).toBe(true);
    expect(config.features.showSteps).toBe(true);
    expect(config.features.showFollowUpSuggestions).toBe(true);
    expect(config.features.showSources).toBe(true);
    expect(config.features.showLatency).toBe(true);
  });

  // =========================================================================
  // UI Configuration
  // =========================================================================

  it("has UI configuration defined", () => {
    expect(config.ui).toBeDefined();
    expect(config.ui.maxMessagesInView).toBeDefined();
    expect(config.ui.animationDuration).toBeDefined();
  });

  it("has correct UI configuration defaults", () => {
    expect(config.ui.maxMessagesInView).toBe(50);
    expect(config.ui.animationDuration).toBe(150);
  });

  it("UI config values are numbers", () => {
    expect(typeof config.ui.maxMessagesInView).toBe("number");
    expect(typeof config.ui.animationDuration).toBe("number");
  });

  it("UI config values are positive", () => {
    expect(config.ui.maxMessagesInView).toBeGreaterThan(0);
    expect(config.ui.animationDuration).toBeGreaterThan(0);
  });

  // =========================================================================
  // Endpoints
  // =========================================================================

  it("has all endpoints defined", () => {
    expect(endpoints.chat).toBeDefined();
    expect(endpoints.chatStream).toBeDefined();
    expect(endpoints.health).toBeDefined();
  });

  it("endpoints are strings", () => {
    expect(typeof endpoints.chat).toBe("string");
    expect(typeof endpoints.chatStream).toBe("string");
    expect(typeof endpoints.health).toBe("string");
  });

  it("endpoints have correct paths", () => {
    expect(endpoints.chat).toContain("/api/chat");
    expect(endpoints.chatStream).toContain("/api/chat/stream");
    expect(endpoints.health).toContain("/api/health");
  });

  it("endpoints use config.apiUrl as base", () => {
    expect(endpoints.chat).toContain(config.apiUrl);
    expect(endpoints.chatStream).toContain(config.apiUrl);
    expect(endpoints.health).toContain(config.apiUrl);
  });

  it("endpoints are valid URL formats", () => {
    expect(endpoints.chat).toMatch(/^https?:\/\/.+\/api\/chat$/);
    expect(endpoints.chatStream).toMatch(/^https?:\/\/.+\/api\/chat\/stream$/);
    expect(endpoints.health).toMatch(/^https?:\/\/.+\/api\/health$/);
  });

  // =========================================================================
  // Example Prompts
  // =========================================================================

  it("has example prompts defined", () => {
    expect(EXAMPLE_PROMPTS).toBeDefined();
    expect(Array.isArray(EXAMPLE_PROMPTS)).toBe(true);
  });

  it("has 5 example prompts", () => {
    expect(EXAMPLE_PROMPTS).toHaveLength(5);
  });

  it("all example prompts are non-empty strings", () => {
    EXAMPLE_PROMPTS.forEach((prompt) => {
      expect(typeof prompt).toBe("string");
      expect(prompt.length).toBeGreaterThan(0);
    });
  });

  it("has expected example prompts", () => {
    expect(EXAMPLE_PROMPTS).toContain(
      "What's going on with Acme Manufacturing in the last 90 days?"
    );
    expect(EXAMPLE_PROMPTS).toContain(
      "Which opportunities are close to renewing this month?"
    );
    expect(EXAMPLE_PROMPTS).toContain("Summarize recent activity for my largest accounts.");
    expect(EXAMPLE_PROMPTS).toContain("Show me the pipeline for TechCorp");
    expect(EXAMPLE_PROMPTS).toContain("What renewals are coming up in the next 30 days?");
  });

  it("all example prompts are unique", () => {
    const uniquePrompts = new Set(EXAMPLE_PROMPTS);
    expect(uniquePrompts.size).toBe(EXAMPLE_PROMPTS.length);
  });

  it("example prompts are reasonably sized", () => {
    EXAMPLE_PROMPTS.forEach((prompt) => {
      expect(prompt.length).toBeGreaterThan(10); // Not too short
      expect(prompt.length).toBeLessThan(200); // Not too long
    });
  });

  // =========================================================================
  // Type Safety and Structure
  // =========================================================================

  it("config has correct type structure", () => {
    expect(typeof config.apiUrl).toBe("string");
    expect(typeof config.features).toBe("object");
    expect(typeof config.ui).toBe("object");
  });

  it("config object is properly structured", () => {
    const configKeys = Object.keys(config);
    expect(configKeys).toContain("apiUrl");
    expect(configKeys).toContain("features");
    expect(configKeys).toContain("ui");
  });

  it("features object has expected keys", () => {
    const featureKeys = Object.keys(config.features);
    expect(featureKeys).toContain("showDataTables");
    expect(featureKeys).toContain("showSteps");
    expect(featureKeys).toContain("showFollowUpSuggestions");
    expect(featureKeys).toContain("showSources");
    expect(featureKeys).toContain("showLatency");
  });

  it("ui object has expected keys", () => {
    const uiKeys = Object.keys(config.ui);
    expect(uiKeys).toContain("maxMessagesInView");
    expect(uiKeys).toContain("animationDuration");
  });

  it("endpoints object has expected keys", () => {
    const endpointKeys = Object.keys(endpoints);
    expect(endpointKeys).toContain("chat");
    expect(endpointKeys).toContain("chatStream");
    expect(endpointKeys).toContain("health");
  });

  // =========================================================================
  // Consistency Checks
  // =========================================================================

  it("all endpoints share same base URL", () => {
    const chatBase = endpoints.chat.replace("/api/chat", "");
    const streamBase = endpoints.chatStream.replace("/api/chat/stream", "");
    const healthBase = endpoints.health.replace("/api/health", "");

    expect(chatBase).toBe(streamBase);
    expect(streamBase).toBe(healthBase);
  });

  it("config is consistent with endpoints", () => {
    expect(endpoints.chat).toContain(config.apiUrl);
    expect(endpoints.chatStream).toContain(config.apiUrl);
    expect(endpoints.health).toContain(config.apiUrl);
  });
});
