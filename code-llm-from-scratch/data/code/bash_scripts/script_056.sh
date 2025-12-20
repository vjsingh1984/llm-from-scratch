#!/bin/bash
# Simple port scanner

HOST="${1:-localhost}"
START_PORT="${2:-1}"
END_PORT="${3:-1024}"

echo "Scanning $HOST ports $START_PORT-$END_PORT"
echo "=========================================="

for port in $(seq $START_PORT $END_PORT); do
    timeout 1 bash -c "echo >/dev/tcp/$HOST/$port" 2>/dev/null && {
        echo "Port $port: OPEN"

        # Try to identify service
        SERVICE=$(getent services $port | awk '{print $1}')
        if [ -n "$SERVICE" ]; then
            echo "  Service: $SERVICE"
        fi
    }
done

echo
echo "Scan complete"
