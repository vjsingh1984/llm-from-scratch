#!/bin/bash
# Analyze error logs for patterns

LOG_FILE="${1:-/var/log/syslog}"
TOP_N="${2:-10}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    exit 1
fi

echo "Error Log Analysis: $LOG_FILE"
echo "=============================="
echo

echo "=== Top $TOP_N Error Messages ==="
grep -i error "$LOG_FILE" |     sed 's/[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}/IP/g' |     sed 's/[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}/DATE/g' |     sort | uniq -c | sort -rn | head -n "$TOP_N"

echo
echo "=== Error Frequency by Hour ==="
grep -i error "$LOG_FILE" |     awk '{print $3}' | cut -d: -f1 | sort | uniq -c

echo
echo "=== Recent Errors (Last 20) ==="
grep -i error "$LOG_FILE" | tail -20
