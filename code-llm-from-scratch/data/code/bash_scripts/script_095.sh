#!/bin/bash
# Database migration during deployment

echo "Running database migrations..."

# Backup
backup_database

# Migrate
run_migrations

# Verify
verify_schema

echo "✓ Migrations complete"
