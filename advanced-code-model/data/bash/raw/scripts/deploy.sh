#!/bin/bash
# Application deployment script

set -euo pipefail

APP_NAME="myapp"
DEPLOY_DIR="/var/www/$APP_NAME"
GIT_REPO="git@github.com:user/${APP_NAME}.git"
BRANCH="${1:-main}"

echo "Deploying $APP_NAME from branch: $BRANCH"

# Backup current version
if [ -d "$DEPLOY_DIR" ]; then
    BACKUP_DIR="/tmp/${APP_NAME}_backup_$(date +%s)"
    echo "Backing up to: $BACKUP_DIR"
    cp -r "$DEPLOY_DIR" "$BACKUP_DIR"
fi

# Clone or pull
if [ ! -d "$DEPLOY_DIR" ]; then
    git clone -b "$BRANCH" "$GIT_REPO" "$DEPLOY_DIR"
else
    cd "$DEPLOY_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
fi

cd "$DEPLOY_DIR"

# Install dependencies
if [ -f "package.json" ]; then
    npm install --production
fi

# Build
if [ -f "Makefile" ]; then
    make build
fi

# Restart service
systemctl restart "$APP_NAME"

echo "Deployment complete!"
