#!/bin/bash
# Database migration runner

MIGRATION_DIR="${1:-migrations}"

echo "Running database migrations from: $MIGRATION_DIR"

for file in "$MIGRATION_DIR"/*.sql; do
    echo "Applying: $(basename "$file")"
    mysql < "$file"
done

echo "✓ Migrations complete"
