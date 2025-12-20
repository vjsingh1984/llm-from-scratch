#!/bin/bash
# Canary deployment validator

CANARY_ENDPOINT="http://canary.example.com"
PRODUCTION_ENDPOINT="http://prod.example.com"
ERROR_THRESHOLD=1

echo "Validating canary deployment..."

canary_errors=$(curl -s "$CANARY_ENDPOINT/metrics" | jq '.error_rate')
prod_errors=$(curl -s "$PRODUCTION_ENDPOINT/metrics" | jq '.error_rate')

if (( $(echo "$canary_errors > $prod_errors + $ERROR_THRESHOLD" | bc -l) )); then
    echo "✗ Canary validation failed"
    echo "Rolling back..."
    exit 1
fi

echo "✓ Canary validated"
