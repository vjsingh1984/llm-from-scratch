#!/bin/bash
# Feature toggle deployment

FEATURE="${1}"
ENABLED="${2:-false}"

echo "Setting feature $FEATURE to $ENABLED"

update_feature_flag "$FEATURE" "$ENABLED"

echo "✓ Feature toggle updated"
