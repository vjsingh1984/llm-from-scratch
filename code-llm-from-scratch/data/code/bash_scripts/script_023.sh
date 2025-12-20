#!/bin/bash
# Simple CI build runner

set -e

PROJECT_DIR="${1:-.}"
BUILD_LOG="build-$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_DIR"

{
    echo "=== CI Build Started ==="
    echo "Date: $(date)"
    echo "Branch: $(git branch --show-current)"
    echo "Commit: $(git rev-parse --short HEAD)"
    echo

    echo "=== Installing Dependencies ==="
    if [ -f "package.json" ]; then
        npm ci
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    elif [ -f "Gemfile" ]; then
        bundle install
    fi

    echo
    echo "=== Running Linter ==="
    if [ -f ".eslintrc.json" ]; then
        npm run lint
    elif [ -f ".pylintrc" ]; then
        pylint src/
    fi

    echo
    echo "=== Running Tests ==="
    if [ -f "package.json" ] && grep -q "test" package.json; then
        npm test
    elif [ -f "pytest.ini" ]; then
        pytest
    elif [ -f "Rakefile" ]; then
        rake test
    fi

    echo
    echo "=== Building ==="
    if [ -f "package.json" ] && grep -q "build" package.json; then
        npm run build
    elif [ -f "setup.py" ]; then
        python setup.py build
    fi

    echo
    echo "=== Build Successful ==="

} 2>&1 | tee "$BUILD_LOG"

exit_code=${PIPESTATUS[0]}

if [ $exit_code -eq 0 ]; then
    echo "✓ Build passed"
else
    echo "✗ Build failed"
fi

exit $exit_code
