# Claude Code Memory

## Standard Post-Implementation Steps

After completing a **significant implementation** (new features, refactors, bug fixes affecting multiple files), execute these steps. Skip for trivial changes (config tweaks, single-line fixes, documentation):

1. **Fix Broken Tests** - Run tests and fix any failures before proceeding
2. **Ensure Coverage ≥ 98%** - Verify test coverage is at least 98%. Add tests if coverage is below threshold
3. **Code Review** - Remove any speculative or dead code. Remove redundant code only when abstraction doesn't add complexity
4. **Run CI** - Execute `./scripts/ci.sh` based on what changed:
   - `./scripts/ci.sh backend` - If only backend/ or tests/backend/ changed
   - `./scripts/ci.sh frontend` - If only frontend/ changed
   - `./scripts/ci.sh` - If both changed or unsure
5. **Run Playwright E2E** - Always run `cd frontend && npm run test:e2e` for end-to-end tests
6. **Push Code** - Stage, commit, and push changes:
   ```bash
   git add -A
   git commit -m "<descriptive commit message>"
   git push
   ```

**Note:** Include these steps in implementation plans for significant work. Use judgment - a one-line fix doesn't need full CI validation.
