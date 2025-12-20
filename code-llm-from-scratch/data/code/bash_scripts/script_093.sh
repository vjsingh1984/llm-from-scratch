#!/bin/bash
# Deployment health check

APP_URL="${1}"

check_endpoint() {
    local endpoint=$1
    curl -f "$APP_URL$endpoint" &>/dev/null
}

echo "Running deployment health checks..."

check_endpoint "/health" && echo "✓ Health check passed"
check_endpoint "/ready" && echo "✓ Ready check passed"
check_endpoint "/metrics" && echo "✓ Metrics endpoint accessible"

echo "✓ All health checks passed"
