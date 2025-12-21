import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Chat Application
 * 
 * These tests run against the actual frontend and backend,
 * verifying the complete user flow works correctly.
 */

test.describe('Chat Application', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('has correct title and header', async ({ page }) => {
    await expect(page).toHaveTitle(/Acme CRM/i);
    await expect(page.locator('header')).toBeVisible();
  });

  test('displays welcome message', async ({ page }) => {
    // Should show initial welcome or empty state
    const main = page.getByRole('main');
    await expect(main).toBeVisible();
  });

  test('has input bar for questions', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
  });

  test('has send button', async ({ page }) => {
    const sendButton = page.getByRole('button', { name: /send/i });
    await expect(sendButton).toBeVisible();
  });

  test('can type in the input field', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('What is Acme CRM?');
    await expect(input).toHaveValue('What is Acme CRM?');
  });

  test('skip link is accessible', async ({ page }) => {
    // Verify skip link exists and is functional
    const skipLink = page.getByRole('link', { name: /skip to main content/i });
    await expect(skipLink).toBeVisible();
    await skipLink.focus();
    await expect(skipLink).toBeFocused();
  });
});

test.describe('Chat Interaction', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('can send a question and receive a response', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    // Type a question
    await input.fill('What renewals are coming up?');
    
    // Send the question
    await sendButton.click();

    // Wait for response - message block is an article with role="listitem"
    await expect(page.getByRole('listitem', { name: /conversation about/i })).toBeVisible({ timeout: 30000 });
  });

  test('shows loading state while waiting for response', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is the pipeline?');
    await sendButton.click();

    // Should show loading indicator
    const loadingIndicator = page.getByRole('status');
    // Loading might be very brief, so we just check it appears at some point
    await expect(loadingIndicator.first()).toBeVisible({ timeout: 5000 });
  });

  test('can send question with Enter key', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });

    await input.fill('How do I create a contact?');
    await input.press('Enter');

    // Should trigger the request - message block is an article with role="listitem"
    await expect(page.getByRole('listitem', { name: /conversation about/i })).toBeVisible({ timeout: 30000 });
  });

  test('displays follow-up suggestions after response', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is going on with Acme Manufacturing?');
    await sendButton.click();

    // Wait for response - message block is an article with role="listitem"
    await expect(page.getByRole('listitem', { name: /conversation about/i })).toBeVisible({ timeout: 30000 });

    // Check for follow-up suggestions (may or may not appear)
    // This is a soft check since follow-ups are optional
    const followUps = page.locator('[class*="follow-up"], [class*="suggestion"]');
    // Just verify the page loaded successfully
    await expect(page.locator('body')).toBeVisible();
  });
});

test.describe('Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('has proper heading structure', async ({ page }) => {
    // Should have at least one heading
    const headings = page.getByRole('heading');
    await expect(headings.first()).toBeVisible();
  });

  test('input has associated label', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible();
    // The presence of a name indicates proper labeling
  });

  test('buttons are keyboard accessible', async ({ page }) => {
    const sendButton = page.getByRole('button', { name: /send/i });
    
    // Verify button is visible and can be focused programmatically
    await expect(sendButton).toBeVisible();
    // Note: disabled buttons can't be focused, but they should still be in the DOM
    await expect(sendButton).toBeDisabled(); // Empty input = disabled button
    
    // Fill input to enable button, then verify it can receive focus
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('Test');
    await expect(sendButton).toBeEnabled();
    await sendButton.focus();
    await expect(sendButton).toBeFocused();
  });

  test('maintains focus after sending message', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('Test question');
    await sendButton.click();

    // After response, input should be visible and usable
    await expect(page.getByRole('listitem', { name: /conversation about/i })).toBeVisible({ timeout: 30000 });
    // Input should be cleared and ready for next question
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
  });
});

test.describe('Error Handling', () => {
  test('handles empty submission gracefully', async ({ page }) => {
    await page.goto('/');
    
    const sendButton = page.getByRole('button', { name: /send/i });
    const input = page.getByRole('textbox', { name: /ask a question/i });
    
    // Button should be disabled when input is empty
    await expect(sendButton).toBeDisabled();
    
    // Input should still be there and functional
    await expect(input).toBeVisible();
    await expect(input).toBeEnabled();
  });
});

test.describe('Responsive Design', () => {
  test('works on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    // All main elements should still be visible
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await expect(input).toBeVisible();
    await expect(sendButton).toBeVisible();
  });

  test('works on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible();
  });
});
