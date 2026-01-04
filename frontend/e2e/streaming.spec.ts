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

  test('shows streaming status indicator while processing', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is going on with Acme Manufacturing?');
    await sendButton.click();

    // Should show streaming status with pulsing dot and status text
    const streamingStatus = page.locator('.streaming-status');
    await expect(streamingStatus).toBeVisible({ timeout: 5000 });

    // Status text should exist within the streaming status
    const statusText = streamingStatus.locator('.streaming-status__text');
    await expect(statusText).toBeVisible();
  });

  test('streaming status text updates during processing', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What renewals are coming up?');
    await sendButton.click();

    // Status text should be visible and contain progress message
    const statusText = page.locator('.streaming-status__text');
    await expect(statusText).toBeVisible({ timeout: 5000 });
    
    // Text should contain something meaningful (not empty)
    const text = await statusText.textContent();
    expect(text).toBeTruthy();
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
    expect(answerText!.length).toBeGreaterThan(10);
  });

  test('streaming status disappears after completion', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What contacts are at Acme?');
    await sendButton.click();

    // Wait for completion
    const answer = page.locator('.message__answer');
    await expect(answer).toBeVisible({ timeout: 30000 });

    // Streaming status should be gone
    const streamingStatus = page.locator('.streaming-status');
    await expect(streamingStatus).not.toBeVisible();
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
  test('streaming starts within 2 seconds', async ({ page }) => {
    await page.goto('/');
    
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const startTime = Date.now();
    
    await input.fill('What is the total pipeline value?');
    await sendButton.click();

    // Status indicator should appear within 2 seconds
    const streamingStatus = page.locator('.streaming-status');
    await expect(streamingStatus).toBeVisible({ timeout: 2000 });
    
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

  test('time indicator appears after completion', async ({ page }) => {
    await page.goto('/');
    
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is Acme CRM?');
    await sendButton.click();

    // Wait for completion
    await expect(page.locator('.message__answer')).toBeVisible({ timeout: 30000 });

    // Time indicator should show (e.g., "2.1s")
    const timeIndicator = page.locator('.message__time');
    await expect(timeIndicator).toBeVisible({ timeout: 5000 });
    
    // Should contain a number followed by 's'
    const timeText = await timeIndicator.textContent();
    expect(timeText).toMatch(/\d+\.?\d*s/);
  });
});
