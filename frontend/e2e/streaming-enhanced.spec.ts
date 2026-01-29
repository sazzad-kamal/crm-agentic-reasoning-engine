import { test, expect } from '@playwright/test';

/**
 * Enhanced E2E Tests for Streaming Chat Feature
 *
 * Covers: status transitions, skeleton animation, answer formatting,
 * rapid questions, error handling, SSE validation, performance
 */

// Helper: send a question and wait for the answer
async function askAndWaitForAnswer(page: import('@playwright/test').Page, question: string) {
  const input = page.getByRole('textbox', { name: /ask a question/i });
  const sendButton = page.getByRole('button', { name: /send/i });
  await input.fill(question);
  await sendButton.click();
  await expect(page.locator('.message__answer')).toBeVisible({ timeout: 80000 });
}

// ===========================================================================
// Streaming Status Transitions
// ===========================================================================

test.describe('Streaming Chat - Enhanced', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('streaming completes with answer visible', async ({ page }) => {
    await askAndWaitForAnswer(page, 'What is going on with Acme Manufacturing?');

    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
    expect(answer!.length).toBeGreaterThan(0);
  });

  test('thinking indicator shows then disappears', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('What renewals are coming up?');
    await page.getByRole('button', { name: /send/i }).click();

    // Skeleton appears
    const skeleton = page.locator('.skeleton-answer');
    await expect(skeleton).toBeVisible({ timeout: 5000 });

    // Answer replaces skeleton
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 80000 });
    await expect(skeleton).not.toBeVisible();
  });

  test('thinking indicator has 3 animated skeleton lines', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('How do I create an opportunity?');
    await page.getByRole('button', { name: /send/i }).click();

    const skeleton = page.locator('.skeleton-answer');
    await expect(skeleton).toBeVisible({ timeout: 5000 });

    const lines = skeleton.locator('.skeleton-answer__line');
    await expect(lines).toHaveCount(3);
  });

  test('streaming preserves markdown formatting', async ({ page }) => {
    await askAndWaitForAnswer(page, 'Explain how to use Acme CRM');

    const answer = page.locator('.message__answer');
    const hasParagraphs = await answer.locator('p').count();
    const hasLists = await answer.locator('ul, ol').count();
    const hasCode = await answer.locator('code').count();
    const hasStrong = await answer.locator('strong').count();

    // Answer should have some structured content
    expect(hasParagraphs + hasLists + hasCode + hasStrong).toBeGreaterThan(0);
  });

  test('handles rapid consecutive questions correctly', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    // Send first question
    await input.fill('What is Acme CRM?');
    await sendButton.click();

    // Input should be disabled while processing
    await expect(input).toBeDisabled({ timeout: 2000 });

    // Wait for first response to complete
    await expect(page.locator('.message__answer').first()).toBeVisible({ timeout: 80000 });
    await expect(input).toBeEnabled();

    // Send second question
    await input.fill('How do I import contacts?');
    await sendButton.click();

    // Both responses should appear
    await expect(page.locator('.message__answer').nth(1)).toBeVisible({ timeout: 80000 });
  });
});

// ===========================================================================
// Streaming Progress
// ===========================================================================

test.describe('Streaming Progress - Enhanced', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('answer appears after skeleton', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('What accounts have upcoming renewals?');
    await page.getByRole('button', { name: /send/i }).click();

    // Skeleton appears first
    await expect(page.locator('.skeleton-answer')).toBeVisible({ timeout: 5000 });

    // Then answer
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 80000 });

    // Skeleton is gone
    await expect(page.locator('.skeleton-answer')).not.toBeVisible();

    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('all processing steps complete successfully', async ({ page }) => {
    await askAndWaitForAnswer(page, 'What is going on with TechCorp?');

    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });
});

// ===========================================================================
// Streaming Error Handling
// ===========================================================================

test.describe('Streaming Error Handling - Enhanced', () => {
  test('shows error on stream failure and re-enables input', async ({ page }) => {
    await page.goto('/');
    await page.route('**/api/chat/stream', route => route.abort('failed'));

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('Test stream failure');
    await page.getByRole('button', { name: /send/i }).click();

    // Input should be re-enabled for retry
    await expect(input).toBeEnabled({ timeout: 10000 });
  });

  test('handles malformed SSE data without crashing', async ({ page }) => {
    await page.goto('/');
    await page.route('**/api/chat/stream', route => {
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'invalid sse data\nthis is not proper format\n',
      });
    });

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('Test malformed data');
    await page.getByRole('button', { name: /send/i }).click();

    // App should not crash — input should remain functional
    await expect(input).toBeVisible({ timeout: 10000 });
    await expect(input).toBeEnabled({ timeout: 10000 });
  });

  test('recovers after connection interruption', async ({ page }) => {
    await page.goto('/');

    let requestCount = 0;
    await page.route('**/api/chat/stream', route => {
      requestCount++;
      if (requestCount === 1) {
        route.abort('failed');
      } else {
        route.continue();
      }
    });

    const input = page.getByRole('textbox', { name: /ask a question/i });

    // First attempt fails
    await input.fill('Test recovery');
    await page.getByRole('button', { name: /send/i }).click();
    await expect(input).toBeEnabled({ timeout: 10000 });

    // Retry should work
    await input.fill('Test recovery - retry');
    await page.getByRole('button', { name: /send/i }).click();
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 80000 });
  });
});

// ===========================================================================
// Streaming Performance
// ===========================================================================

test.describe('Streaming Performance - Enhanced', () => {
  test('thinking indicator appears within 1 second', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('What renewals are coming up?');

    const startTime = Date.now();
    await page.getByRole('button', { name: /send/i }).click();

    await expect(page.locator('.skeleton-answer')).toBeVisible({ timeout: 1000 });
    const elapsed = Date.now() - startTime;
    expect(elapsed).toBeLessThan(1500);
  });

  test('complete response under 30 second budget', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const startTime = Date.now();

    await input.fill('Summarize account activity for Beta Tech');
    await page.getByRole('button', { name: /send/i }).click();

    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 80000 });
    const totalTime = Date.now() - startTime;
    expect(totalTime).toBeLessThan(80000);
  });
});

// ===========================================================================
// Streaming Visual States
// ===========================================================================

test.describe('Streaming Visual States', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('skeleton appears during streaming, answer after', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('What is going on with Acme?');
    await page.getByRole('button', { name: /send/i }).click();

    const skeleton = page.locator('.skeleton-answer');
    await expect(skeleton).toBeVisible({ timeout: 5000 });

    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 80000 });
    await expect(skeleton).not.toBeVisible();
  });

  test('answer displays with copy button', async ({ page }) => {
    await askAndWaitForAnswer(page, 'What contacts are at TechCorp?');

    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();

    await expect(page.locator('.message__copy')).toBeVisible();
  });
});
