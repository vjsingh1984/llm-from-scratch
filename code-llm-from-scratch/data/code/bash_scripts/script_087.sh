#!/bin/bash
# Rolling deployment script

SERVERS=("app1" "app2" "app3")
NEW_VERSION="${1}"
DEPLOY_SCRIPT="/opt/deploy.sh"

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

echo "Rolling Deployment"
echo "=================="
echo "Version: $NEW_VERSION"
echo "Servers: ${SERVERS[*]}"
echo

deploy_to_server() {
    local server=$1

    echo "Deploying to $server..."

    # Remove from load balancer
    echo "  Draining $server..."
    ssh "$server" "touch /var/www/maintenance"
    sleep 10

    # Deploy
    echo "  Updating application..."
    ssh "$server" "$DEPLOY_SCRIPT $NEW_VERSION"

    # Health check
    echo "  Running health check..."
    for i in {1..10}; do
        if ssh "$server" "curl -f http://localhost/health" &>/dev/null; then
            echo "  ✓ Health check passed"

            # Add back to load balancer
            ssh "$server" "rm /var/www/maintenance"

            echo "  ✓ $server deployment complete"
            return 0
        fi
        sleep 2
    done

    echo "  ✗ Health check failed"
    return 1
}

for server in "${SERVERS[@]}"; do
    if ! deploy_to_server "$server"; then
        echo "✗ Deployment failed on $server"
        echo "Aborting rolling deployment"
        exit 1
    fi

    echo
    sleep 5
done

echo "✓ Rolling deployment complete"
