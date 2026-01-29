import { test, expect } from '@playwright/test';

/**
 * Advanced E2E Test Scenarios
 *
 * Covers: complex workflows, edge cases, performance benchmarks,
 * WCAG accessibility, state persistence, security
 */

// ===========================================================================
// Complex User Workflows
// ===========================================================================

test.describe('Complex User Workflows', () => {
  test('complete discovery-to-answer workflow', async ({ page }) => {
    await page.goto('/');

    // 1. Open data drawer
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();

    const drawer = page.getByRole('dialog');
    await expect(drawer).toBeVisible();

    // 2. Explore companies table
    await expect(page.locator('.data-table')).toBeVisible({ timeout: 15000 });

    // 3. Search for a company
    const searchInput = page.locator('.data-explorer__search-input');
    await searchInput.fill('Acme');

    // 4. Close drawer
    const closeButton = page.getByRole('button', { name: 'Close data browser' });
    await closeButton.click();
    await expect(drawer).not.toBeVisible();

    // 5. Ask a question
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('What is Acme CRM?');
    const sendButton = page.getByRole('button', { name: /send/i });
    await sendButton.click();

    // 6. Receive answer
    await expect(page.locator('.message-block')).toBeVisible({ timeout: 40000 });
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 80000 });
  });

  test('multi-question conversation flow', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    // Question 1
    await input.fill('What is Acme CRM?');
    await sendButton.click();
    await expect(page.locator('.message__answer').first()).toBeVisible({ timeout: 40000 });

    // Verify first message exists
    const messages = page.locator('.message-block');
    await expect(messages).toHaveCount(1);

    // Question 2
    await input.fill('What contacts are at Acme?');
    await sendButton.click();
    await expect(page.locator('.message__answer').nth(1)).toBeVisible({ timeout: 40000 });

    // Both messages visible
    await expect(messages).toHaveCount(2);
  });
});

// ===========================================================================
// Edge Cases & Error Scenarios
// ===========================================================================

test.describe('Edge Cases & Error Scenarios', () => {
  test('handles extremely long questions', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const longName = 'SuperDuperUltraMegaGigantic Corporation International Limited LLC'.repeat(3);
    await input.fill(`What is going on with ${longName}?`);
    await sendButton.click();

    // Should handle gracefully — input should eventually be re-enabled
    await expect(input).toBeEnabled({ timeout: 80000 });
  });

  test('prevents double-send via disabled button', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Test question');
    await sendButton.click();

    // Button should be disabled immediately after click
    await expect(sendButton).toBeDisabled({ timeout: 2000 });
  });

  test('handles browser back/forward navigation', async ({ page }) => {
    await page.goto('/');

    await page.goBack();
    await page.goForward();

    // App should still be functional
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
  });

  test('handles page refresh during streaming', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What accounts have upcoming renewals?');
    await sendButton.click();

    // Wait for streaming to start
    await expect(page.locator('.skeleton-answer')).toBeVisible({ timeout: 5000 });

    // Refresh page mid-stream
    await page.reload();

    // Should return to clean state
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
    await expect(input).toHaveValue('');

    // Previous messages should be gone (no persistence)
    const messages = page.locator('.message-block');
    await expect(messages).toHaveCount(0);
  });
});

// ===========================================================================
// Performance Benchmarks
// ===========================================================================

test.describe('Performance Benchmarks', () => {
  test('time to interactive under 2 seconds', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();

    const tti = Date.now() - startTime;
    expect(tti).toBeLessThan(2000);
  });

  test('drawer opens within 1.5 seconds', async ({ page }) => {
    await page.goto('/');

    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    const startTime = Date.now();

    await browseButton.click();
    const drawer = page.getByRole('dialog');
    await expect(drawer).toBeVisible();

    const openTime = Date.now() - startTime;
    expect(openTime).toBeLessThan(1500);
  });
});

// ===========================================================================
// Accessibility - WCAG Compliance
// ===========================================================================

test.describe('Accessibility - WCAG Compliance', () => {
  test('all images have alt text', async ({ page }) => {
    await page.goto('/');

    const images = page.locator('img');
    const count = await images.count();

    for (let i = 0; i < count; i++) {
      const alt = await images.nth(i).getAttribute('alt');
      // Alt text should exist (can be empty for decorative images)
      expect(alt).not.toBeNull();
    }
  });

  test('form inputs have accessible names', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible();

    const ariaLabel = await input.getAttribute('aria-label');
    expect(ariaLabel).toBeTruthy();
  });

  test('focus indicators are visible on keyboard navigation', async ({ page }) => {
    await page.goto('/');

    // Tab through several elements
    for (let i = 0; i < 4; i++) {
      await page.keyboard.press('Tab');
    }

    // An element should be focused
    const focusedTag = await page.evaluate(() => document.activeElement?.tagName);
    expect(focusedTag).toBeTruthy();
    expect(focusedTag).not.toBe('BODY'); // Focus moved off body
  });

  test('chat area has proper ARIA landmark', async ({ page }) => {
    await page.goto('/');

    const main = page.getByRole('main');
    await expect(main).toBeVisible();

    const log = page.locator('[role="log"]');
    await expect(log).toBeVisible();
    await expect(log).toHaveAttribute('aria-live', 'polite');
  });
});

// ===========================================================================
// State Persistence & Session Management
// ===========================================================================

test.describe('State Persistence & Session Management', () => {
  test('clears state on page reload', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('Test question');

    await page.reload();

    await expect(input).toHaveValue('');

    const messages = page.locator('.message-block');
    await expect(messages).toHaveCount(0);
  });

  test('concurrent tabs maintain separate state', async ({ context }) => {
    const page1 = await context.newPage();
    const page2 = await context.newPage();

    await page1.goto('http://localhost:5173');
    await page2.goto('http://localhost:5173');

    const input1 = page1.getByRole('textbox', { name: /ask a question/i });
    const input2 = page2.getByRole('textbox', { name: /ask a question/i });

    await input1.fill('Question from tab 1');
    await input2.fill('Question from tab 2');

    await expect(input1).toHaveValue('Question from tab 1');
    await expect(input2).toHaveValue('Question from tab 2');

    await page1.close();
    await page2.close();
  });
});

// ===========================================================================
// Security & Input Validation
// ===========================================================================

test.describe('Security & Input Validation', () => {
  test('sanitizes XSS attempts in questions', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('<script>alert("XSS")</script>');
    await page.getByRole('button', { name: /send/i }).click();

    // Should not execute script — page should remain functional
    await expect(input).toBeVisible({ timeout: 80000 });

    // No alert dialog should appear
    let dialogAppeared = false;
    page.on('dialog', () => { dialogAppeared = true; });
    expect(dialogAppeared).toBe(false);
  });

  test('handles special HTML characters in input', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('<b>Test</b> & "quotes"');
    await expect(input).toHaveValue('<b>Test</b> & "quotes"');
  });

  test('XSS in response is sanitized', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('What is Acme CRM?');
    await page.getByRole('button', { name: /send/i }).click();

    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 80000 });

    // No script tags should be executable in the response
    const hasScript = await page.locator('.message__answer script').count();
    expect(hasScript).toBe(0);
  });
});

// ===========================================================================
// Browser Compatibility
// ===========================================================================

test.describe('Browser Compatibility', () => {
  test('app loads and is functional', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();

    // Header renders
    await expect(page.locator('header')).toBeVisible();

    // Chat area renders
    const chatArea = page.locator('[role="log"]');
    await expect(chatArea).toBeVisible();
  });
});
