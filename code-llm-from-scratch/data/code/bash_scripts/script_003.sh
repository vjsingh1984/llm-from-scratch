#!/bin/bash
# Service health checker with email alerts

SERVICES=("nginx" "mysql" "redis")
EMAIL="admin@example.com"
FAILED_SERVICES=()

check_service() {
    local service=$1

    if systemctl is-active --quiet "$service"; then
        echo "✓ $service is running"
        return 0
    else
        echo "✗ $service is NOT running"
        FAILED_SERVICES+=("$service")
        return 1
    fi
}

restart_service() {
    local service=$1
    echo "Attempting to restart $service..."

    systemctl restart "$service"

    if systemctl is-active --quiet "$service"; then
        echo "✓ $service restarted successfully"
        return 0
    else
        echo "✗ Failed to restart $service"
        return 1
    fi
}

send_alert() {
    local message=$1
    echo "$message" | mail -s "Service Alert" "$EMAIL"
}

echo "Checking services..."

for service in "${SERVICES[@]}"; do
    if ! check_service "$service"; then
        restart_service "$service" || {
            send_alert "$service failed and could not be restarted"
        }
    fi
done

if [ ${#FAILED_SERVICES[@]} -gt 0 ]; then
    echo "Failed services: ${FAILED_SERVICES[*]}"
    exit 1
fi

echo "All services healthy"
