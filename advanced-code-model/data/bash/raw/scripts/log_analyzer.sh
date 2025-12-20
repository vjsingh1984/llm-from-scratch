#!/bin/bash
# Log file analyzer

LOG_FILE="${1:-/var/log/syslog}"
SEARCH_TERM="${2:-error}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "Analyzing log file: $LOG_FILE"
echo "Search term: $SEARCH_TERM"
echo "================================"
echo ""

# Count occurrences
COUNT=$(grep -i "$SEARCH_TERM" "$LOG_FILE" | wc -l)
echo "Total matches: $COUNT"
echo ""

# Show recent matches
echo "Recent matches (last 10):"
grep -i "$SEARCH_TERM" "$LOG_FILE" | tail -10
echo ""

# Hourly distribution
echo "Hourly distribution (last 24 hours):"
grep -i "$SEARCH_TERM" "$LOG_FILE" | awk '{print $3}' | cut -d: -f1 | sort | uniq -c | sort -rn
