#!/bin/bash
# Feature flag manager

FLAG_NAME="${1}"
FLAG_VALUE="${2}"
FLAGS_FILE="/etc/app/feature-flags.json"

echo "Setting feature flag: $FLAG_NAME=$FLAG_VALUE"

# Update flag
jq ".$FLAG_NAME = $FLAG_VALUE" "$FLAGS_FILE" > "$FLAGS_FILE.tmp"
mv "$FLAGS_FILE.tmp" "$FLAGS_FILE"

# Reload app
systemctl reload app

echo "✓ Feature flag updated"
