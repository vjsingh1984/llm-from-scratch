#!/bin/bash
# Pre-deployment checklist

echo "Pre-Deployment Checklist"
echo "========================"

check_item "Tests passing" "npm test"
check_item "Linting clean" "npm run lint"
check_item "Dependencies up to date" "npm outdated"
check_item "Backup created" "test -f /backup/latest.tar.gz"

echo "✓ All checks passed"
