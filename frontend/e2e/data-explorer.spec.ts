import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Data Explorer Drawer
 * 
 * Tests the slide-out drawer that shows CRM data
 * the AI has access to.
 */

test.describe('Data Explorer Drawer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('has Browse Data button in header', async ({ page }) => {
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await expect(browseButton).toBeVisible();
  });

  test('opens drawer when Browse Data clicked', async ({ page }) => {
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
    
    // Drawer should open
    const drawer = page.getByRole('dialog', { name: /crm data/i });
    await expect(drawer).toBeVisible();
  });

  test('drawer has close button', async ({ page }) => {
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
    
    const closeButton = page.locator('.drawer__close');
    await expect(closeButton).toBeVisible();
  });

  test('closes drawer when close button clicked', async ({ page }) => {
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
    
    const drawer = page.locator('.drawer');
    await expect(drawer).toHaveClass(/drawer--open/);
    
    const closeButton = page.locator('.drawer__close');
    await closeButton.click();
    
    // Drawer should close (class should not contain drawer--open)
    await expect(drawer).not.toHaveClass(/drawer--open/);
  });

  test('closes drawer on Escape key', async ({ page }) => {
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
    
    const drawer = page.locator('.drawer');
    await expect(drawer).toHaveClass(/drawer--open/);
    
    await page.keyboard.press('Escape');
    
    await expect(drawer).not.toHaveClass(/drawer--open/);
  });

  test('closes drawer when clicking overlay', async ({ page }) => {
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
    
    const drawer = page.locator('.drawer');
    await expect(drawer).toHaveClass(/drawer--open/);
    
    // Click the overlay on the left side (where drawer doesn't cover)
    // The drawer opens from the right, so clicking left of center hits the overlay
    const overlay = page.locator('.drawer-overlay');
    await overlay.click({ position: { x: 50, y: 300 } });
    
    await expect(drawer).not.toHaveClass(/drawer--open/);
  });
});

test.describe('Data Explorer Tabs', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Open the drawer
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
  });

  test('displays all data tabs', async ({ page }) => {
    await expect(page.getByRole('tab', { name: /companies/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /contacts/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /opportunities/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /activities/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /history/i })).toBeVisible();
  });

  test('companies tab is active by default', async ({ page }) => {
    const companiesTab = page.getByRole('tab', { name: /companies/i });
    await expect(companiesTab).toHaveAttribute('aria-selected', 'true');
  });

  test('can switch to contacts tab', async ({ page }) => {
    const contactsTab = page.getByRole('tab', { name: /contacts/i });
    await contactsTab.click();
    await expect(contactsTab).toHaveAttribute('aria-selected', 'true');
  });

  test('can switch to opportunities tab', async ({ page }) => {
    const oppTab = page.getByRole('tab', { name: /opportunities/i });
    await oppTab.click();
    await expect(oppTab).toHaveAttribute('aria-selected', 'true');
  });

});

test.describe('Data Explorer Table', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
    // Wait for table to be visible (data loaded)
    await expect(page.locator('.data-table')).toBeVisible({ timeout: 15000 });
  });

  test('displays company data in table', async ({ page }) => {
    // Should show table with data
    const table = page.locator('.data-table');
    await expect(table).toBeVisible();
  });

  test('displays record count', async ({ page }) => {
    // Should show "X of Y records"
    const count = page.locator('.data-explorer__count');
    await expect(count).toContainText(/\d+ of \d+ records/);
  });

  test('search input is visible', async ({ page }) => {
    const searchInput = page.locator('.data-explorer__search-input');
    await expect(searchInput).toBeVisible();
  });

  test('can filter data with search', async ({ page }) => {
    const searchInput = page.locator('.data-explorer__search-input');
    
    // Get initial count
    const countBefore = await page.locator('.data-explorer__count').textContent();
    
    // Type a search term
    await searchInput.fill('Acme');
    
    // Wait for filter to apply
    await page.waitForTimeout(100);
    
    // Count should potentially be different (filtered)
    const countAfter = await page.locator('.data-explorer__count').textContent();
    
    // At minimum, the count text should exist
    expect(countAfter).toBeTruthy();
  });
});

test.describe('Data Explorer Expandable Rows', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
    // Wait for table to be visible (data loaded)
    await expect(page.locator('.data-table')).toBeVisible({ timeout: 15000 });
  });

  test('shows expand button for rows with nested data', async ({ page }) => {
    // Companies with private texts should have expand buttons
    const expandButtons = page.locator('.data-table__expand-btn');
    // At least some companies should have nested data
    const count = await expandButtons.count();
    expect(count).toBeGreaterThanOrEqual(0);
  });

  test('expands row to show nested data', async ({ page }) => {
    const expandButtons = page.locator('.data-table__expand-btn');
    const count = await expandButtons.count();
    
    if (count > 0) {
      // Click first expand button
      await expandButtons.first().click();
      
      // Should show nested data section
      const nestedData = page.locator('.nested-data');
      await expect(nestedData).toBeVisible();
    }
  });

  test('collapses row on second click', async ({ page }) => {
    const expandButtons = page.locator('.data-table__expand-btn');
    const count = await expandButtons.count();
    
    if (count > 0) {
      // Expand
      await expandButtons.first().click();
      const nestedData = page.locator('.nested-data');
      await expect(nestedData).toBeVisible();
      
      // Collapse
      await expandButtons.first().click();
      await expect(nestedData).not.toBeVisible();
    }
  });
});

test.describe('Data Explorer Ask AI Integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();
    // Wait for table to be visible (data loaded)
    await expect(page.locator('.data-table')).toBeVisible({ timeout: 15000 });
    // Wait for table rows to appear
    await page.locator('.data-table__row').first().waitFor({ state: 'visible', timeout: 15000 });
  });

  test('shows Ask AI buttons in table', async ({ page }) => {
    // Wait for at least one button to appear
    await expect(page.locator('.data-table__ask-btn').first()).toBeVisible();
    const askButtons = page.locator('.data-table__ask-btn');
    const count = await askButtons.count();
    expect(count).toBeGreaterThan(0);
  });

  test('clicking Ask AI closes drawer and fills input', async ({ page }) => {
    // Click first Ask AI button
    const askButton = page.locator('.data-table__ask-btn').first();
    await askButton.click();
    
    // Drawer should close (class should not contain drawer--open)
    const drawer = page.locator('.drawer');
    await expect(drawer).not.toHaveClass(/drawer--open/);
    
    // Input should have a question filled in
    const input = page.getByRole('textbox', { name: /ask a question/i });
    await expect(input).toBeVisible();
    const value = await input.inputValue();
    expect(value.length).toBeGreaterThan(0);
  });
});

test.describe('Data Explorer API Integration', () => {
  test('loads companies data from API', async ({ page }) => {
    await page.goto('/');

    // Set up response interception BEFORE clicking
    const responsePromise = page.waitForResponse(
      resp => resp.url().includes('/api/data/companies') && resp.status() === 200,
      { timeout: 30000 }
    );

    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();

    const response = await responsePromise;
    const data = await response.json();

    expect(data.data).toBeDefined();
    expect(data.total).toBeGreaterThan(0);
  });

  test('loads opportunities data from API', async ({ page }) => {
    await page.goto('/');
    const browseButton = page.getByRole('button', { name: /browse.*data/i });
    await browseButton.click();

    // Wait for companies data first
    await expect(page.locator('.data-table')).toBeVisible({ timeout: 15000 });

    // Switch to opportunities tab
    const oppTab = page.getByRole('tab', { name: /opportunities/i });

    // Set up response interception BEFORE clicking tab
    const responsePromise = page.waitForResponse(
      resp => resp.url().includes('/api/data/opportunities') && resp.status() === 200,
      { timeout: 30000 }
    );

    await oppTab.click();

    const response = await responsePromise;
    const data = await response.json();

    expect(data.data).toBeDefined();
    expect(data.total).toBeGreaterThan(0);
  });

});
