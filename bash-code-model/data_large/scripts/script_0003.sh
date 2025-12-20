#!/bin/bash
# Advanced log analyzer

LOG_FILE=${1:-"/var/log/syslog"}
OUTPUT_DIR="./log_analysis"

mkdir -p "$OUTPUT_DIR"

echo "Analyzing $LOG_FILE..."

# Error count
echo "--- Error Summary ---" > "$OUTPUT_DIR/errors.txt"
grep -i "error" "$LOG_FILE" | cut -d':' -f4- | sort | uniq -c | sort -rn >> "$OUTPUT_DIR/errors.txt"

# Warning count
echo "--- Warning Summary ---" > "$OUTPUT_DIR/warnings.txt"
grep -i "warning" "$LOG_FILE" | cut -d':' -f4- | sort | uniq -c | sort -rn >> "$OUTPUT_DIR/warnings.txt"

# Timeline
echo "--- Hourly Activity ---" > "$OUTPUT_DIR/timeline.txt"
awk '{print $3}' "$LOG_FILE" | cut -d':' -f1 | sort | uniq -c >> "$OUTPUT_DIR/timeline.txt"

echo "Analysis complete. Results in $OUTPUT_DIR/"
