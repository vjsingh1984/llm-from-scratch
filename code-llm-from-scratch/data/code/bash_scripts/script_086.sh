#!/bin/bash
# Blue-green deployment script

APP_NAME="myapp"
NEW_VERSION="$1"
BLUE_PORT=8080
GREEN_PORT=8081
LB_CONFIG="/etc/nginx/sites-enabled/lb.conf"

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

# Detect current active environment
CURRENT_PORT=$(grep -oP 'proxy_pass.*:\K\d+' "$LB_CONFIG")

if [ "$CURRENT_PORT" = "$BLUE_PORT" ]; then
    INACTIVE_PORT=$GREEN_PORT
    INACTIVE_ENV="green"
else
    INACTIVE_PORT=$BLUE_PORT
    INACTIVE_ENV="blue"
fi

echo "Current: port $CURRENT_PORT"
echo "Deploying to: $INACTIVE_ENV (port $INACTIVE_PORT)"

# Deploy to inactive environment
deploy_to_inactive() {
    docker pull "$APP_NAME:$NEW_VERSION"

    docker stop "$APP_NAME-$INACTIVE_ENV" 2>/dev/null
    docker rm "$APP_NAME-$INACTIVE_ENV" 2>/dev/null

    docker run -d         --name "$APP_NAME-$INACTIVE_ENV"         -p "$INACTIVE_PORT:8080"         "$APP_NAME:$NEW_VERSION"

    # Health check
    for i in {1..30}; do
        if curl -f "http://localhost:$INACTIVE_PORT/health" >/dev/null 2>&1; then
            echo "✓ Health check passed"
            return 0
        fi
        sleep 2
    done

    echo "✗ Health check failed"
    return 1
}

switch_traffic() {
    # Update load balancer
    sed -i "s/proxy_pass.*:$CURRENT_PORT/proxy_pass http:\/\/localhost:$INACTIVE_PORT/" "$LB_CONFIG"

    # Reload nginx
    nginx -t && nginx -s reload

    echo "✓ Traffic switched to $INACTIVE_ENV"
}

if deploy_to_inactive; then
    switch_traffic
    echo "Deployment successful!"
else
    echo "Deployment failed!"
    exit 1
fi
