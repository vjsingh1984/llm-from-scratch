#!/bin/bash
# Database vacuum (PostgreSQL)

DB_NAME="${1:-postgres}"

echo "Running VACUUM ANALYZE on: $DB_NAME"

psql -d "$DB_NAME" -c "VACUUM ANALYZE VERBOSE;"

echo "✓ Vacuum complete"
