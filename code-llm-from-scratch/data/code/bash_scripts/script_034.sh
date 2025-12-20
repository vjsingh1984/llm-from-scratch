#!/bin/bash
# Configuration drift detector

EXPECTED_CONFIG="/etc/app/config.expected.json"
ACTUAL_CONFIG="/etc/app/config.json"

echo "Checking configuration drift..."

diff -u "$EXPECTED_CONFIG" "$ACTUAL_CONFIG" || {
    echo "⚠ Configuration drift detected"
    exit 1
}

echo "✓ No drift"
