#!/bin/bash
# Configuration management

ENV="${1}"
CONFIG_FILE="/etc/app/config-${ENV}.yml"

echo "Applying configuration for: $ENV"

cp "$CONFIG_FILE" /etc/app/config.yml
systemctl reload app

echo "✓ Configuration applied"
