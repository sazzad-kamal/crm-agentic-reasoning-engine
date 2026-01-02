#!/bin/bash
# Local CI script - runs all checks before pushing to GitHub
# Usage: ./scripts/ci.sh [backend|frontend|all]

set -e  # Exit on first error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

run_backend() {
    print_header "Backend: Ruff Linter"
    ruff check backend/ --select=E,F,W --ignore=E501,F401,F841,E402,E701,E741
    print_success "Ruff linter passed"

    print_header "Backend: Ruff Formatter"
    ruff format backend/ --check --diff
    print_success "Ruff formatter passed"

    print_header "Backend: Mypy Type Check"
    mypy backend/ --ignore-missing-imports --no-error-summary --allow-untyped-defs
    print_success "Mypy passed"

    print_header "Backend: Pytest"
    MOCK_LLM=1 OPENAI_API_KEY=test-key pytest tests/backend/ -v --tb=short
    print_success "Backend tests passed"
}

run_frontend() {
    print_header "Frontend: ESLint"
    (cd frontend && npm run lint)
    print_success "ESLint passed"

    print_header "Frontend: TypeScript"
    (cd frontend && npx tsc --noEmit)
    print_success "TypeScript passed"

    print_header "Frontend: Vitest"
    (cd frontend && npm test)
    print_success "Frontend tests passed"

    print_header "Frontend: Build"
    (cd frontend && npm run build)
    print_success "Frontend build passed"
}

# Parse arguments
TARGET="${1:-all}"

echo -e "${GREEN}Running local CI checks...${NC}"

case $TARGET in
    backend)
        run_backend
        ;;
    frontend)
        run_frontend
        ;;
    all)
        run_backend
        run_frontend
        ;;
    *)
        echo "Usage: $0 [backend|frontend|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All CI checks passed!${NC}"
echo -e "${GREEN}========================================${NC}"
