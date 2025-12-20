#!/bin/bash
# Simple website deployment script

REPO_URL="${1}"
DEPLOY_DIR="/var/www/html"
SERVICE_NAME="nginx"

if [ -z "$REPO_URL" ]; then
    echo "Usage: $0 <git-repo-url>"
    exit 1
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Cloning repository..."
git clone "$REPO_URL" .

# Run tests if available
if [ -f "package.json" ]; then
    echo "Running tests..."
    npm install
    npm test || { echo "Tests failed!"; exit 1; }
fi

# Backup current deployment
if [ -d "$DEPLOY_DIR" ]; then
    echo "Backing up current deployment..."
    tar -czf "/tmp/backup_$(date +%Y%m%d_%H%M%S).tar.gz" "$DEPLOY_DIR"
fi

# Deploy
echo "Deploying..."
rsync -av --delete "$TEMP_DIR/" "$DEPLOY_DIR/"

# Restart service
echo "Restarting $SERVICE_NAME..."
systemctl restart "$SERVICE_NAME"

# Cleanup
cd /
rm -rf "$TEMP_DIR"

echo "Deployment complete!"
