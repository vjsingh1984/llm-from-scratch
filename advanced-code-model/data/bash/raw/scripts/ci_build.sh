#!/bin/bash
# Continuous Integration build script

set -euo pipefail

PROJECT_NAME="myproject"
BUILD_DIR="build"

echo "CI Build Started"
echo "================"
echo "Project: $PROJECT_NAME"
echo "Branch: ${GIT_BRANCH:-main}"
echo "Commit: ${GIT_COMMIT:-$(git rev-parse HEAD)}"
echo ""

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

# Install dependencies
echo "Installing dependencies..."
if [ -f "package.json" ]; then
    npm ci
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Run linting
echo "Running linters..."
if [ -f "package.json" ]; then
    npm run lint || true
fi

# Run tests
echo "Running tests..."
if [ -f "package.json" ]; then
    npm test
elif [ -f "Makefile" ]; then
    make test
fi

# Build
echo "Building..."
if [ -f "package.json" ]; then
    npm run build
elif [ -f "Makefile" ]; then
    make build
fi

echo ""
echo "Build successful!"
