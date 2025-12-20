#!/bin/bash
# Automated git deployment script

REPO_URL="${1:-}"
DEPLOY_DIR="/var/www/app"
BRANCH="${2:-main}"

if [ -z "$REPO_URL" ]; then
    echo "Usage: $0 <repo-url> [branch]"
    exit 1
fi

deploy() {
    echo "Deploying from $REPO_URL (branch: $BRANCH)"

    if [ -d "$DEPLOY_DIR/.git" ]; then
        cd "$DEPLOY_DIR"
        git fetch origin
        git reset --hard "origin/$BRANCH"
    else
        git clone -b "$BRANCH" "$REPO_URL" "$DEPLOY_DIR"
    fi

    cd "$DEPLOY_DIR"

    # Install dependencies
    if [ -f "package.json" ]; then
        npm install --production
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi

    # Restart service
    systemctl restart app

    echo "Deployment complete"
}

# Backup current deployment
if [ -d "$DEPLOY_DIR" ]; then
    cp -r "$DEPLOY_DIR" "$DEPLOY_DIR.backup.$(date +%Y%m%d_%H%M%S)"
fi

deploy || {
    echo "Deployment failed!"
    if [ -d "$DEPLOY_DIR.backup."* ]; then
        echo "Restoring backup..."
        rm -rf "$DEPLOY_DIR"
        mv "$DEPLOY_DIR.backup."* "$DEPLOY_DIR"
    fi
    exit 1
}
