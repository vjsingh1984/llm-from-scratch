#!/bin/bash
# Find largest files and directories

SEARCH_DIR="${1:-.}"
TOP_N="${2:-10}"

echo "Finding top $TOP_N largest files in: $SEARCH_DIR"
echo "=================================================="
echo

echo "Largest Files:"
echo "-------------"
find "$SEARCH_DIR" -type f -exec du -h {} + 2>/dev/null |     sort -rh | head -n "$TOP_N" |     awk '{printf "%-10s %s
", $1, $2}'

echo
echo "Largest Directories:"
echo "-------------------"
du -h "$SEARCH_DIR"/* 2>/dev/null |     sort -rh | head -n "$TOP_N" |     awk '{printf "%-10s %s
", $1, $2}'

echo
echo "Disk Usage Summary:"
echo "------------------"
df -h "$SEARCH_DIR"
