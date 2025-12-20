#!/bin/bash
# Aggregate logs from multiple sources

OUTPUT_DIR="/var/log/aggregated"
DATE=$(date +%Y%m%d)

mkdir -p "$OUTPUT_DIR"

aggregate_logs() {
    echo "Aggregating logs for $(date)"

    # System logs
    {
        echo "=== System Logs ==="
        tail -n 1000 /var/log/syslog
    } > "$OUTPUT_DIR/system-$DATE.log"

    # Application logs
    {
        echo "=== Application Logs ==="
        find /var/log/app -name "*.log" -exec tail -n 100 {} \;
    } > "$OUTPUT_DIR/app-$DATE.log"

    # Web server logs
    {
        echo "=== Web Server Logs ==="
        tail -n 1000 /var/log/nginx/access.log
        tail -n 1000 /var/log/nginx/error.log
    } > "$OUTPUT_DIR/web-$DATE.log"

    # Database logs
    {
        echo "=== Database Logs ==="
        tail -n 1000 /var/log/mysql/error.log
    } > "$OUTPUT_DIR/db-$DATE.log"

    echo "✓ Logs aggregated to: $OUTPUT_DIR"
}

compress_old_logs() {
    echo "Compressing old logs..."

    find "$OUTPUT_DIR" -name "*.log" -mtime +1 ! -name "*.gz" -exec gzip {} \;
    find "$OUTPUT_DIR" -name "*.gz" -mtime +30 -delete

    echo "✓ Old logs compressed"
}

aggregate_logs
compress_old_logs
