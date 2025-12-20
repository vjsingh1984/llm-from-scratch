#!/bin/bash
# Canary deployment

NEW_VERSION="${1}"
CANARY_PERCENT=10

echo "Deploying canary: $CANARY_PERCENT% traffic to $NEW_VERSION"

# Deploy canary
deploy_canary "$NEW_VERSION" "$CANARY_PERCENT"

# Monitor for 10 minutes
sleep 600

# Check error rates
if check_error_rate_acceptable; then
    echo "✓ Canary successful, rolling out to 100%"
    deploy_full "$NEW_VERSION"
else
    echo "✗ Canary failed, rolling back"
    rollback_canary
fi
