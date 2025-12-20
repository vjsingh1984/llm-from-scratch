#!/bin/bash
# Deployment notification

ENVIRONMENT="${1}"
VERSION="${2}"
STATUS="${3}"

MESSAGE="Deployment to $ENVIRONMENT: $VERSION - $STATUS"

# Slack notification
curl -X POST -H 'Content-type: application/json'     --data "{"text":"$MESSAGE"}"     "$SLACK_WEBHOOK_URL"

echo "✓ Notification sent"
