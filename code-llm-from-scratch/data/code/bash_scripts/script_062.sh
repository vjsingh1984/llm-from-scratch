#!/bin/bash
# Network latency monitor

TARGET="${1:-8.8.8.8}"

echo "Monitoring latency to $TARGET..."

ping "$TARGET" | while read line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $line"
done
