#!/bin/bash
# Sync environment variables between environments

SOURCE_ENV="${1}"
TARGET_ENV="${2}"

if [ -z "$SOURCE_ENV" ] || [ -z "$TARGET_ENV" ]; then
    echo "Usage: $0 <source-env-file> <target-env-file>"
    exit 1
fi

if [ ! -f "$SOURCE_ENV" ]; then
    echo "Error: Source file not found: $SOURCE_ENV"
    exit 1
fi

echo "Environment Sync"
echo "================"
echo "Source: $SOURCE_ENV"
echo "Target: $TARGET_ENV"
echo

# Backup target if it exists
if [ -f "$TARGET_ENV" ]; then
    cp "$TARGET_ENV" "${TARGET_ENV}.backup-$(date +%Y%m%d_%H%M%S)"
    echo "✓ Backed up target file"
fi

# Create target if it doesn't exist
touch "$TARGET_ENV"

# Read source and update target
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ "$key" =~ ^#.*$ ]] || [ -z "$key" ] && continue

    # Remove existing key from target
    sed -i "/^$key=/d" "$TARGET_ENV"

    # Prompt for value
    echo -n "[$key] Current: $value, New value (Enter to keep): "
    read new_value

    if [ -z "$new_value" ]; then
        new_value="$value"
    fi

    # Add to target
    echo "$key=$new_value" >> "$TARGET_ENV"

done < "$SOURCE_ENV"

echo
echo "✓ Environment sync complete"
echo "Updated: $TARGET_ENV"
