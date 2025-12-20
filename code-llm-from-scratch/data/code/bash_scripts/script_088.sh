#!/bin/bash
# Complete deployment pipeline

set -e

PROJECT_DIR="${1:-.}"
ENVIRONMENT="${2:-staging}"

cd "$PROJECT_DIR"

echo "Deployment Pipeline"
echo "==================="
echo "Environment: $ENVIRONMENT"
echo "Commit: $(git rev-parse --short HEAD)"
echo

stage_build() {
    echo "=== Stage 1: Build ==="

    if [ -f "package.json" ]; then
        npm ci
        npm run build
    elif [ -f "Dockerfile" ]; then
        docker build -t "app:$ENVIRONMENT" .
    fi

    echo "✓ Build complete"
}

stage_test() {
    echo
    echo "=== Stage 2: Test ==="

    if [ -f "package.json" ]; then
        npm test
    elif [ -f "pytest.ini" ]; then
        pytest
    fi

    echo "✓ Tests passed"
}

stage_deploy() {
    echo
    echo "=== Stage 3: Deploy ==="

    if [ "$ENVIRONMENT" = "production" ]; then
        echo "⚠ Deploying to PRODUCTION"
        read -p "Continue? (yes/no) " -r
        [ "$REPLY" != "yes" ] && exit 1
    fi

    # Deploy based on environment
    ./deploy.sh "$ENVIRONMENT"

    echo "✓ Deployment complete"
}

stage_verify() {
    echo
    echo "=== Stage 4: Verification ==="

    APP_URL=$(get_app_url "$ENVIRONMENT")

    for i in {1..10}; do
        if curl -f "$APP_URL/health" &>/dev/null; then
            echo "✓ Application is healthy"
            return 0
        fi
        sleep 5
    done

    echo "✗ Verification failed"
    return 1
}

# Run pipeline
stage_build
stage_test
stage_deploy
stage_verify

echo
echo "✓ Pipeline complete"
