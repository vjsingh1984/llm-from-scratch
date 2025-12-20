#!/bin/bash
# Trending analyzer

LOG_FILE="/var/log/metrics.log"

echo "7-Day Trend Analysis"
echo "===================="

for day in {1..7}; do
    date_str=$(date -d "$day days ago" +%Y-%m-%d)
    count=$(grep "$date_str" "$LOG_FILE" | wc -l)
    echo "$date_str: $count events"
done
