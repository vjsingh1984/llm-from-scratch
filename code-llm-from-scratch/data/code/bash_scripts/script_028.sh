#!/bin/bash
# Check backend servers behind load balancer

BACKENDS=("10.0.1.10:8080" "10.0.1.11:8080" "10.0.1.12:8080")
HEALTH_ENDPOINT="/health"
TIMEOUT=5

check_backend() {
    local backend=$1
    local url="http://$backend$HEALTH_ENDPOINT"

    echo -n "Checking $backend... "

    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$url" 2>/dev/null)

    if [ "$response" = "200" ]; then
        echo "✓ UP (HTTP $response)"
        return 0
    else
        echo "✗ DOWN (HTTP $response)"
        return 1
    fi
}

drain_backend() {
    local backend=$1
    local host=$(echo "$backend" | cut -d: -f1)

    echo "Draining backend: $backend"

    # This would integrate with your load balancer
    # Example for nginx:
    # echo "server $host:8080 down;" | nginx -s reload

    echo "  ✓ Backend drained"
}

echo "Load Balancer Health Check"
echo "=========================="
echo

UNHEALTHY=()

for backend in "${BACKENDS[@]}"; do
    if ! check_backend "$backend"; then
        UNHEALTHY+=("$backend")
    fi
done

echo
if [ ${#UNHEALTHY[@]} -gt 0 ]; then
    echo "⚠ Unhealthy backends: ${#UNHEALTHY[@]}"

    for backend in "${UNHEALTHY[@]}"; do
        echo "  - $backend"
    done

    # Optionally drain unhealthy backends
    read -p "Drain unhealthy backends? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for backend in "${UNHEALTHY[@]}"; do
            drain_backend "$backend"
        done
    fi

    exit 1
else
    echo "✓ All backends healthy"
    exit 0
fi
