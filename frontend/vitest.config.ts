import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "happy-dom",
    globals: true,
    setupFiles: ["./src/__tests__/setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html", "lcov"],
      exclude: [
        "**/__tests__/**",
        "**/*.test.{ts,tsx}",
        "**/*.spec.{ts,tsx}",
        "**/node_modules/**",
        "**/dist/**",
        "**/types/**",
        "**/*.d.ts",
        "**/main.tsx",
        "**/vite-env.d.ts",
        "**/*.css",
        "**/components/index.ts",
        "**/components/dataExplorer/index.ts",
        "**/components/dataExplorer/types.ts",
      ],
      thresholds: {
        statements: 98,
        branches: 90,
        functions: 99,
        lines: 99,
      },
    },
    // Timeout settings
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
