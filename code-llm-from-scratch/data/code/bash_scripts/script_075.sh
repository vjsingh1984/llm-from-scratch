#!/bin/bash
# Monitor application uptime

APP_URL="${1:-http://localhost}"
CHECK_INTERVAL="${2:-60}"
LOG_FILE="/var/log/uptime-monitor.log"

echo "Monitoring: $APP_URL"
echo "Interval: ${CHECK_INTERVAL}s"
echo "Log: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo

check_uptime() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$APP_URL")

    if [ "$response" = "200" ]; then
        echo "[$timestamp] ✓ UP (HTTP $response)" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[$timestamp] ✗ DOWN (HTTP $response)" | tee -a "$LOG_FILE"

        # Send alert
        echo "Application down: $APP_URL" | mail -s "Uptime Alert" admin@example.com

        return 1
    fi
}

while true; do
    check_uptime
    sleep "$CHECK_INTERVAL"
done
