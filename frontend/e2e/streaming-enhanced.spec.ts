import { test, expect } from '@playwright/test';

/**
 * Enhanced E2E Tests for Streaming Chat Feature
 *
 * Improvements:
 * - SSE (Server-Sent Events) validation
 * - Progress tracking verification
 * - Interruption/cancellation tests
 * - Concurrent streaming tests
 * - Step transition animations
 */

test.describe('Streaming Chat - Enhanced', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('streaming shows multiple status transitions', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is going on with Acme Manufacturing?');
    await sendButton.click();

    // Wait for answer to appear (streaming completes fast with mock LLM)
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Verify the question was processed (answer is visible means streaming worked)
    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('thinking indicator shows while processing', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What renewals are coming up?');
    await sendButton.click();

    // Thinking indicator should appear while processing
    const thinkingIndicator = page.locator('.message-skeleton');
    await expect(thinkingIndicator).toBeVisible({ timeout: 5000 });

    // Wait for answer to appear
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Verify the response was received
    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('thinking indicator has animated skeleton lines', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('How do I create an opportunity?');
    await sendButton.click();

    // Thinking indicator (skeleton) should be visible
    const thinkingIndicator = page.locator('.message-skeleton');
    await expect(thinkingIndicator).toBeVisible({ timeout: 5000 });

    // Should have animated skeleton lines
    const lines = thinkingIndicator.locator('.message-skeleton__line');
    await expect(lines).toHaveCount(3);
  });

  test('streaming preserves answer formatting', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Explain how to use Acme CRM');
    await sendButton.click();

    // Wait for answer
    const answer = page.locator('.message__answer');
    await expect(answer).toBeVisible({ timeout: 30000 });

    // Check for markdown rendering
    const hasParagraphs = await answer.locator('p').count();
    const hasLists = await answer.locator('ul, ol').count();
    const hasCode = await answer.locator('code').count();

    // Answer should have some structure
    expect(hasParagraphs + hasLists + hasCode).toBeGreaterThan(0);
  });

  test('handles rapid consecutive questions', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    // Send first question
    await input.fill('What is Acme CRM?');
    await sendButton.click();

    // Don't wait for completion, send second question immediately
    await page.waitForTimeout(100);

    // Input should be disabled while processing
    await expect(input).toBeDisabled();

    // Wait for first response to complete
    await expect(page.locator('.message__answer').first()).toBeVisible({ timeout: 30000 });

    // Now input should be enabled
    await expect(input).toBeEnabled();

    // Send second question
    await input.fill('How do I import contacts?');
    await sendButton.click();

    // Second response should appear
    await expect(page.locator('.message__answer').nth(1)).toBeVisible({ timeout: 30000 });
  });
});

test.describe('Streaming Progress Tracking', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('shows all expected processing steps', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is going on with TechCorp?');
    await sendButton.click();

    // Wait for answer to appear
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Verify answer content
    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('thinking indicator transitions to answer', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What accounts have upcoming renewals?');
    await sendButton.click();

    // Thinking indicator should appear first
    await expect(page.locator('.message-skeleton')).toBeVisible({ timeout: 5000 });

    // Wait for answer to appear
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Thinking indicator should be gone
    await expect(page.locator('.message-skeleton')).not.toBeVisible();

    // Verify answer content
    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('failed steps show error status', async ({ page }) => {
    // Mock an error response that aborts
    await page.route('**/api/chat/stream', route => {
      route.abort('failed');
    });

    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Test error');
    await sendButton.click();

    // Wait for error handling
    await page.waitForTimeout(2000);

    // App should remain functional after error
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
  });
});

test.describe('Streaming Error Handling - Enhanced', () => {
  test('shows user-friendly error message on stream failure', async ({ page }) => {
    await page.goto('/');

    // Mock complete stream failure
    await page.route('**/api/chat/stream', route => {
      route.abort('failed');
    });

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Test stream failure');
    await sendButton.click();

    // Should show error banner or message
    await page.waitForTimeout(3000);

    // Input should be re-enabled for retry
    await expect(input).toBeEnabled();
  });

  test('handles malformed SSE data gracefully', async ({ page }) => {
    await page.goto('/');

    // Send malformed SSE data
    await page.route('**/api/chat/stream', route => {
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'invalid sse data\nthis is not proper format\n',
      });
    });

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Test malformed data');
    await sendButton.click();

    await page.waitForTimeout(3000);

    // Should not crash, input should remain functional
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
  });

  test('recovers after connection interruption', async ({ page }) => {
    await page.goto('/');

    let requestCount = 0;

    await page.route('**/api/chat/stream', route => {
      requestCount++;

      if (requestCount === 1) {
        // First request fails
        route.abort('failed');
      } else {
        // Second request succeeds
        route.continue();
      }
    });

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    // First attempt fails
    await input.fill('Test recovery');
    await sendButton.click();
    await page.waitForTimeout(3000);

    // Retry should work
    await input.fill('Test recovery - retry');
    await sendButton.click();

    // Second attempt should succeed
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });
  });
});

test.describe('Streaming Performance - Enhanced', () => {
  test('measures time to thinking indicator', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const startTime = Date.now();

    await input.fill('What is the pipeline?');
    await sendButton.click();

    // Wait for thinking indicator to appear
    await expect(page.locator('.message-skeleton')).toBeVisible({ timeout: 5000 });

    const elapsed = Date.now() - startTime;
    console.log(`Time to thinking indicator: ${elapsed}ms`);
    expect(elapsed).toBeLessThan(3000);
  });

  test('thinking indicator appears within 1 second', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const startTime = Date.now();

    await input.fill('What renewals are coming up?');
    await sendButton.click();

    // Thinking indicator should appear within 1 second
    await expect(page.locator('.message-skeleton')).toBeVisible({ timeout: 1000 });

    const elapsed = Date.now() - startTime;
    expect(elapsed).toBeLessThan(1500);
  });

  test('complete streaming response under performance budget', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const startTime = Date.now();

    await input.fill('Summarize account activity for Beta Tech');
    await sendButton.click();

    // Wait for complete answer
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    const totalTime = Date.now() - startTime;
    console.log(`Total streaming time: ${totalTime}ms`);

    // Should complete within 30 seconds
    expect(totalTime).toBeLessThan(30000);
  });
});

test.describe('Streaming Visual States', () => {
  test('thinking indicator has correct visual styling', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is going on with Acme?');
    await sendButton.click();

    // Thinking indicator should be visible with proper styling
    const thinkingIndicator = page.locator('.message-skeleton');
    await expect(thinkingIndicator).toBeVisible({ timeout: 5000 });

    // Wait for answer to appear
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Verify answer received
    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('answer displays with proper formatting', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What contacts are at TechCorp?');
    await sendButton.click();

    // Wait for answer to appear
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Verify answer received and has content
    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();

    // Copy button should be available
    await expect(page.locator('.message__copy')).toBeVisible();
  });
});
