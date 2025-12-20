#!/bin/bash
# Custom log rotation script for application logs

APP_LOG_DIR="/var/log/myapp"
ROTATE_DAYS=7
COMPRESS_AFTER_DAYS=1
ARCHIVE_DIR="/var/log/myapp/archive"

mkdir -p "$ARCHIVE_DIR"

rotate_log() {
    local logfile=$1
    local basename=$(basename "$logfile")

    if [ ! -f "$logfile" ]; then
        return
    fi

    # Rotate if file exists and has content
    if [ -s "$logfile" ]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local rotated="${ARCHIVE_DIR}/${basename}.${timestamp}"

        echo "Rotating: $logfile -> $rotated"
        cp "$logfile" "$rotated"
        > "$logfile"

        # Compress old rotated logs
        find "$ARCHIVE_DIR" -name "${basename}.*" -mtime +$COMPRESS_AFTER_DAYS ! -name "*.gz" -exec gzip {} \;

        # Delete old archives
        find "$ARCHIVE_DIR" -name "${basename}.*.gz" -mtime +$ROTATE_DAYS -delete
    fi
}

# Rotate all application logs
for logfile in "$APP_LOG_DIR"/*.log; do
    rotate_log "$logfile"
done

echo "Log rotation complete"
