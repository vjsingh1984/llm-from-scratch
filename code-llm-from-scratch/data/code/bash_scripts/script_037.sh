#!/bin/bash
# Dependency version checker

PROJECT_FILE="${1:-package.json}"

echo "Checking dependencies for updates..."

if [ -f "package.json" ]; then
    npm outdated
elif [ -f "requirements.txt" ]; then
    pip list --outdated
elif [ -f "Gemfile" ]; then
    bundle outdated
fi
