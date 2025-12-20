#!/bin/bash
# Automated disk cleanup script
# Removes old logs, temp files, and cached data

LOG_DIRS=("/var/log" "/tmp")
DAYS_OLD=30
DRY_RUN=false

usage() {
    echo "Usage: $0 [-d days] [-n (dry-run)]"
    exit 1
}

while getopts "d:n" opt; do
    case $opt in
        d) DAYS_OLD=$OPTARG ;;
        n) DRY_RUN=true ;;
        *) usage ;;
    esac
done

echo "Disk Cleanup Script"
echo "==================="
echo "Cleaning files older than $DAYS_OLD days"
echo "Dry run: $DRY_RUN"
echo

for dir in "${LOG_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Cleaning: $dir"

        if [ "$DRY_RUN" = true ]; then
            find "$dir" -type f -mtime +$DAYS_OLD -print
        else
            find "$dir" -type f -mtime +$DAYS_OLD -delete
        fi
    fi
done

# Clean package manager cache
if [ "$DRY_RUN" = false ]; then
    apt-get clean 2>/dev/null || yum clean all 2>/dev/null
fi

echo "Cleanup complete"
df -h
