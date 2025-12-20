#!/bin/bash
# Monitor critical system files for changes

WATCH_DIRS=("/etc" "/boot" "/usr/bin")
BASELINE_FILE="/var/lib/fim/baseline.db"
REPORT_FILE="/var/lib/fim/changes-$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$(dirname "$BASELINE_FILE")"

create_baseline() {
    echo "Creating baseline..."

    {
        for dir in "${WATCH_DIRS[@]}"; do
            find "$dir" -type f -exec sha256sum {} \; 2>/dev/null
        done
    } > "$BASELINE_FILE"

    echo "Baseline created: $BASELINE_FILE"
    echo "Total files: $(wc -l < "$BASELINE_FILE")"
}

check_integrity() {
    if [ ! -f "$BASELINE_FILE" ]; then
        echo "Error: No baseline found. Run with --create-baseline first"
        exit 1
    fi

    echo "Checking file integrity..."
    echo "==========================" > "$REPORT_FILE"

    CHANGES=0
    ADDED=0
    REMOVED=0
    MODIFIED=0

    # Create current snapshot
    TEMP_SNAPSHOT="/tmp/fim-snapshot-$$.db"
    {
        for dir in "${WATCH_DIRS[@]}"; do
            find "$dir" -type f -exec sha256sum {} \; 2>/dev/null
        done
    } > "$TEMP_SNAPSHOT"

    # Compare with baseline
    echo "Modified files:" >> "$REPORT_FILE"
    while IFS= read -r line; do
        hash=$(echo "$line" | awk '{print $1}')
        file=$(echo "$line" | cut -d' ' -f3-)

        baseline_hash=$(grep " $file$" "$BASELINE_FILE" | awk '{print $1}')

        if [ -z "$baseline_hash" ]; then
            echo "  [ADDED] $file" >> "$REPORT_FILE"
            ((ADDED++))
            ((CHANGES++))
        elif [ "$hash" != "$baseline_hash" ]; then
            echo "  [MODIFIED] $file" >> "$REPORT_FILE"
            ((MODIFIED++))
            ((CHANGES++))
        fi
    done < "$TEMP_SNAPSHOT"

    # Check for removed files
    echo >> "$REPORT_FILE"
    echo "Removed files:" >> "$REPORT_FILE"
    while IFS= read -r line; do
        file=$(echo "$line" | cut -d' ' -f3-)

        if ! grep -q " $file$" "$TEMP_SNAPSHOT"; then
            echo "  [REMOVED] $file" >> "$REPORT_FILE"
            ((REMOVED++))
            ((CHANGES++))
        fi
    done < "$BASELINE_FILE"

    # Summary
    echo >> "$REPORT_FILE"
    echo "Summary:" >> "$REPORT_FILE"
    echo "  Added: $ADDED" >> "$REPORT_FILE"
    echo "  Modified: $MODIFIED" >> "$REPORT_FILE"
    echo "  Removed: $REMOVED" >> "$REPORT_FILE"
    echo "  Total changes: $CHANGES" >> "$REPORT_FILE"

    cat "$REPORT_FILE"

    rm -f "$TEMP_SNAPSHOT"

    if [ $CHANGES -gt 0 ]; then
        echo
        echo "⚠ Changes detected! Report: $REPORT_FILE"
        exit 1
    else
        echo
        echo "✓ No changes detected"
        rm -f "$REPORT_FILE"
        exit 0
    fi
}

case "${1:-check}" in
    --create-baseline)
        create_baseline
        ;;
    --check|check)
        check_integrity
        ;;
    *)
        echo "Usage: $0 [--create-baseline|--check]"
        exit 1
        ;;
esac
