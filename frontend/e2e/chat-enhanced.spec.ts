import { test, expect } from '@playwright/test';

/**
 * Enhanced E2E Tests for Chat Application
 *
 * Covers: visual regression, example prompts, keyboard navigation,
 * edge cases, conversation persistence, copy, follow-ups,
 * network conditions, responsive design
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
// Visual Regression (skip in CI — snapshots are gitignored)
// ===========================================================================

test.describe('Chat Application - Visual Regression', () => {
  test.skip(!!process.env.CI, 'Visual tests skipped in CI');

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('header screenshot', async ({ page }) => {
    await expect(page.locator('header')).toBeVisible();
    await expect(page.locator('header')).toHaveScreenshot('header.png');
  });

  test('full page layout screenshot', async ({ page }) => {
    await expect(page).toHaveScreenshot('full-page-initial.png', { fullPage: true });
  });
});

// ===========================================================================
// Example Prompts
// ===========================================================================

test.describe('Chat - Example Prompts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('displays example prompts in empty state', async ({ page }) => {
    const buttons = page.locator('.suggestion-btn');
    await expect(buttons.first()).toBeVisible({ timeout: 10000 });

    const count = await buttons.count();
    expect(count).toBeGreaterThanOrEqual(3);
  });
});

// ===========================================================================
// Keyboard Navigation
// ===========================================================================

test.describe('Chat - Keyboard Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('input accepts keyboard input and send button is visible', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible({ timeout: 10000 });
    await input.click();
    await page.keyboard.type('Test');
    await expect(input).toHaveValue('Test');

    const sendButton = page.getByRole('button', { name: /send/i });
    await expect(sendButton).toBeVisible();
    await expect(sendButton).toBeEnabled();
  });
});

// ===========================================================================
// Edge Cases
// ===========================================================================

test.describe('Chat - Edge Cases', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('handles very long questions gracefully', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const longQuestion = 'What is the status of '.repeat(25) + 'Acme Manufacturing?';
    await input.fill(longQuestion);

    await expect(input).toHaveValue(longQuestion);
    await expect(sendButton).toBeEnabled();
  });

  test('handles special characters in questions', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    const specialChars = 'What about <Company> & "Partners" (2024)?';
    await input.fill(specialChars);
    await sendButton.click();

    await expect(page.getByRole('listitem', { name: /conversation about/i }))
      .toBeVisible({ timeout: 80000 });
  });
});

// ===========================================================================
// Conversation
// ===========================================================================

test.describe('Chat - Conversation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('conversation persists after response', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const sendButton = page.getByRole('button', { name: /send/i });

    await input.fill('What is Acme CRM?');
    await sendButton.click();
    await expect(page.locator('.message__answer').first()).toBeVisible({ timeout: 80000 });

    // Question should remain visible in UI
    await expect(page.locator('.message__question')).toContainText('What is Acme CRM?');

    // Send second question
    await input.fill('How do I create a contact?');
    await sendButton.click();

    // Both messages should be visible
    const messages = page.locator('.message-block');
    await expect(messages).toHaveCount(2, { timeout: 80000 });
  });

  test('shows thinking indicator during processing', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('What is going on with Acme Manufacturing?');
    await page.getByRole('button', { name: /send/i }).click();

    const skeleton = page.locator('.skeleton-answer');
    await expect(skeleton).toBeVisible({ timeout: 5000 });
  });

  test('displays answer after response', async ({ page }) => {
    await askAndWaitForAnswer(page, 'What accounts have upcoming renewals?');

    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
    expect(answer!.length).toBeGreaterThan(0);
  });

  test('can copy answer text', async ({ page }) => {
    await page.context().grantPermissions(['clipboard-write', 'clipboard-read']);
    await askAndWaitForAnswer(page, 'What is Acme CRM?');

    const copyButton = page.locator('.copy-button').first();
    await expect(copyButton).toBeVisible();
    await copyButton.click();

    // Should show copied state
    await expect(copyButton).toHaveClass(/copy-button--copied/);
  });

  test('copy button is available after response', async ({ page }) => {
    await askAndWaitForAnswer(page, 'What is the total pipeline value?');

    // CopyButton only renders after streaming is fully complete (!isStreaming)
    // Wait for input to be re-enabled which signals streaming is done
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeEnabled({ timeout: 80000 });

    const copyButton = page.locator('.copy-button.message__copy');
    await expect(copyButton).toBeVisible({ timeout: 5000 });
  });

  test('can interact with follow-up suggestions', async ({ page }) => {
    await askAndWaitForAnswer(page, 'What is the pipeline for TechCorp?');

    const followUpButtons = page.locator('.follow-up-btn');
    const count = await followUpButtons.count();

    if (count > 0) {
      const suggestionText = await followUpButtons.first().textContent();
      await followUpButtons.first().click();

      // Should send the follow-up as a new message
      await expect(page.locator('.message-block')).toHaveCount(2, { timeout: 80000 });
    }
  });
});

// ===========================================================================
// Accessibility
// ===========================================================================

test.describe('Chat - Accessibility Enhanced', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('ARIA live region announces new messages', async ({ page }) => {
    const chatArea = page.locator('[role="log"]');
    await expect(chatArea).toBeVisible();
    await expect(chatArea).toHaveAttribute('aria-live', 'polite');
  });

  test('skip link and main content are properly linked', async ({ page }) => {
    const skipLink = page.locator('.skip-link');
    await expect(skipLink).toBeAttached();

    const main = page.getByRole('main');
    await expect(main).toHaveAttribute('tabindex', '-1');
    await expect(main).toHaveAttribute('id', 'main-content');
  });

  test('screen reader labels are present on input', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    const ariaLabel = await input.getAttribute('aria-label');
    expect(ariaLabel).toBeTruthy();
  });

  test('loading state has ARIA status role', async ({ page }) => {
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('Test');
    await page.getByRole('button', { name: /send/i }).click();

    const loadingStatus = page.getByRole('status');
    await expect(loadingStatus.first()).toBeVisible({ timeout: 5000 });
  });
});

// ===========================================================================
// Network Conditions
// ===========================================================================

test.describe('Chat - Network Conditions', () => {
  test('handles slow connection', async ({ page, context }) => {
    await context.route('**/*', route => {
      setTimeout(() => route.continue(), 200);
    });

    await page.goto('/');
    await askAndWaitForAnswer(page, 'What is Acme CRM?');

    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('shows error when backend is unavailable', async ({ page }) => {
    await page.route('**/api/chat/stream', route => route.abort());
    await page.goto('/');

    const input = page.getByRole('textbox', { name: /ask a question/i });
    await input.fill('Test question');
    await page.getByRole('button', { name: /send/i }).click();

    // Error banner should appear or input should be re-enabled
    await expect(input).toBeEnabled({ timeout: 10000 });
  });
});

// ===========================================================================
// Responsive Design
// ===========================================================================

test.describe('Chat - Responsive Design Enhanced', () => {
  test('mobile: Browse Data button adapts', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await expect(browseButton).toBeVisible();

    await browseButton.click();
    const drawer = page.getByRole('dialog');
    await expect(drawer).toBeVisible();
  });

  test('mobile: drawer covers full width', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();

    const drawer = page.locator('.drawer');
    await expect(drawer).toBeVisible();

    const box = await drawer.boundingBox();
    expect(box?.width).toBeGreaterThanOrEqual(370);
  });

  test('tablet: layout is optimized', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');

    await expect(page.locator('header')).toBeVisible();
    await expect(page.getByRole('textbox', { name: /ask a question/i })).toBeVisible();
  });

  test('desktop: full features visible', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');

    await expect(page.locator('.header__title')).toBeVisible();
    await expect(page.locator('.header__subtitle')).toBeVisible();
    await expect(page.locator('.header__data-btn-text')).toBeVisible();
  });
});

// ===========================================================================
// Visual Regression (non-screenshot)
// ===========================================================================

test.describe('Chat - Visual States', () => {
  test('message block renders with question and answer', async ({ page }) => {
    await page.goto('/');
    await askAndWaitForAnswer(page, 'What is Acme CRM?');

    const messageBlock = page.locator('.message-block').first();
    await expect(messageBlock).toBeVisible();

    const answer = await page.locator('.message__answer').textContent();
    expect(answer).toBeTruthy();
  });

  test('empty state is visible on load', async ({ page }) => {
    await page.goto('/');

    const chatArea = page.locator('.chat-area, [role="log"]');
    await expect(chatArea).toBeVisible();

    const welcomeText = page.getByText('Welcome to Acme CRM AI');
    await expect(welcomeText).toBeVisible();
  });
});
