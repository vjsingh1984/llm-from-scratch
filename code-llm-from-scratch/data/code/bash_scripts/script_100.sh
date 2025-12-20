#!/bin/bash
# Post-deployment verification

APP_URL="${1}"

echo "Post-Deployment Verification"
echo "============================"

# Check health
curl -f "$APP_URL/health"

# Run smoke tests
./smoke-tests.sh "$APP_URL"

# Verify database
verify_database_connections

echo "✓ Verification complete"
