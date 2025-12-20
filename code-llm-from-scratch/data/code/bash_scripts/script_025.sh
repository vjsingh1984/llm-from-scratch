#!/bin/bash
# Docker container health checker

check_container() {
    local container=$1

    echo "Checking: $container"

    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^$container$"; then
        echo "  ✗ Container is not running"
        return 1
    fi

    # Check health status
    health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null)

    if [ "$health" = "healthy" ]; then
        echo "  ✓ Status: healthy"
        return 0
    elif [ "$health" = "unhealthy" ]; then
        echo "  ✗ Status: unhealthy"

        # Show recent health check logs
        echo "  Recent health checks:"
        docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' "$container" | tail -5

        return 1
    else
        # No health check defined
        echo "  ⚠ No health check configured"
        return 0
    fi
}

restart_container() {
    local container=$1

    echo "Restarting container: $container"
    docker restart "$container"

    sleep 5

    if check_container "$container"; then
        echo "  ✓ Container restarted successfully"
        return 0
    else
        echo "  ✗ Container still unhealthy after restart"
        return 1
    fi
}

echo "Docker Container Health Check"
echo "============================="
echo

UNHEALTHY=()

# Check all running containers
for container in $(docker ps --format '{{.Names}}'); do
    if ! check_container "$container"; then
        UNHEALTHY+=("$container")
    fi
    echo
done

# Try to restart unhealthy containers
if [ ${#UNHEALTHY[@]} -gt 0 ]; then
    echo "Found ${#UNHEALTHY[@]} unhealthy container(s)"
    echo

    for container in "${UNHEALTHY[@]}"; do
        restart_container "$container"
        echo
    done
fi
