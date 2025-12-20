#!/bin/bash
# Zero-downtime deployment

echo "Starting zero-downtime deployment..."

# Start new version alongside old
start_new_version

# Gradually shift traffic
shift_traffic 25
sleep 60
shift_traffic 50
sleep 60
shift_traffic 75
sleep 60
shift_traffic 100

# Stop old version
stop_old_version

echo "✓ Zero-downtime deployment complete"
