#!/bin/bash
# Service health dashboard

SERVICES=("nginx" "mysql" "redis")

echo "Service Health Dashboard"
echo "======================="

for service in "${SERVICES[@]}"; do
    if systemctl is-active --quiet "$service"; then
        echo "✓ $service: running"
    else
        echo "✗ $service: stopped"
    fi
done
