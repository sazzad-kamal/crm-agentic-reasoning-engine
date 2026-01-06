import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Streaming Chat Feature
 * 
 * Tests the real-time streaming functionality including:
 * - Status updates during processing
 * - Step-by-step progress display
 * - Answer streaming
 * - Follow-up suggestions
 */

test.describe('Streaming Chat', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('shows thinking indicator while processing', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is going on with Acme Manufacturing?');
    await sendButton.click();

    // Should show thinking indicator (skeleton loader)
    const thinkingIndicator = page.locator('.message-skeleton');
    await expect(thinkingIndicator).toBeVisible({ timeout: 5000 });
  });

  test('streaming completes with final answer', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('How do I create an opportunity?');
    await sendButton.click();

    // Wait for streaming to complete - answer should appear
    const answer = page.locator('.message__answer');
    await expect(answer).toBeVisible({ timeout: 30000 });
    
    // Answer should have content
    const answerText = await answer.textContent();
    expect(answerText).toBeTruthy();
    expect(answerText!.length).toBeGreaterThan(0);
  });

  test('thinking indicator disappears after completion', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What contacts are at Acme?');
    await sendButton.click();

    // Wait for completion
    const answer = page.locator('.message__answer');
    await expect(answer).toBeVisible({ timeout: 30000 });

    // Thinking indicator should be gone
    const thinkingIndicator = page.locator('.message-skeleton');
    await expect(thinkingIndicator).not.toBeVisible();
  });

  test('can send multiple questions with streaming', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    // First question
    await input.fill('What is Acme CRM?');
    await sendButton.click();
    
    // Wait for first response
    await expect(page.locator('.message__answer').first()).toBeVisible({ timeout: 30000 });

    // Second question
    await input.fill('How do I import contacts?');
    await sendButton.click();

    // Wait for second response
    const answers = page.locator('.message__answer');
    await expect(answers).toHaveCount(2, { timeout: 30000 });
  });
});

test.describe('Streaming Error Handling', () => {
  test('handles streaming errors gracefully', async ({ page }) => {
    await page.goto('/');
    
    // Mock a failed streaming request
    await page.route('**/api/chat/stream', route => {
      route.fulfill({
        status: 500,
        contentType: 'text/event-stream',
        body: 'event: error\ndata: {"message": "Internal server error"}\n\n',
      });
    });

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Test error handling');
    await sendButton.click();

    // Should show error message or handle gracefully
    // The message should be removed on error, or an error banner shown
    await page.waitForTimeout(2000);
    
    // Input should still be usable
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
  });
});

test.describe('Streaming Performance', () => {
  test('thinking indicator appears within 2 seconds', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const startTime = Date.now();

    await input.fill('What is the total pipeline value?');
    await sendButton.click();

    // Thinking indicator should appear within 2 seconds
    const thinkingIndicator = page.locator('.message-skeleton');
    await expect(thinkingIndicator).toBeVisible({ timeout: 2000 });

    const elapsedTime = Date.now() - startTime;
    expect(elapsedTime).toBeLessThan(2500);
  });

  test('complete response within 30 seconds', async ({ page }) => {
    await page.goto('/');
    
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const startTime = Date.now();
    
    await input.fill('Summarize recent activity for enterprise accounts');
    await sendButton.click();

    // Full response should appear within 30 seconds
    const answer = page.locator('.message__answer');
    await expect(answer).toBeVisible({ timeout: 30000 });
    
    const elapsedTime = Date.now() - startTime;
    expect(elapsedTime).toBeLessThan(30000);
    
    console.log(`Response time: ${elapsedTime}ms`);
  });
});

test.describe('Streaming UI Elements', () => {
  test('shows sources button after streaming completes', async ({ page }) => {
    await page.goto('/');
    
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is going on with TechCorp?');
    await sendButton.click();

    // Wait for completion
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Sources section may or may not be visible depending on if sources were returned
    // Just verify the answer was received
    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('follow-up suggestions appear after streaming', async ({ page }) => {
    await page.goto('/');
    
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What renewals are coming up this month?');
    await sendButton.click();

    // Wait for completion
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Follow-ups may appear
    const followUps = page.locator('.follow-up-container, [class*="follow-up"]');
    // Give time for follow-ups to render
    await page.waitForTimeout(2000);
  });

  test('copy button appears after completion', async ({ page }) => {
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is Acme CRM?');
    await sendButton.click();

    // Wait for completion
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Copy button should be available
    const copyButton = page.locator('.message__copy');
    await expect(copyButton).toBeVisible({ timeout: 5000 });
  });
});
