#!/bin/bash
# Rotate secrets and update services

SECRETS_FILE="/etc/secrets/app.env"
SERVICES=("webapp" "api" "worker")

rotate_secret() {
    local key=$1

    # Generate new secret
    local new_secret=$(openssl rand -base64 32)

    echo "Rotating: $key"

    # Update secrets file
    sed -i "s/^$key=.*/$key=$new_secret/" "$SECRETS_FILE"

    echo "  ✓ Updated secrets file"

    return 0
}

reload_services() {
    echo
    echo "Reloading services..."

    for service in "${SERVICES[@]}"; do
        echo "  Restarting $service..."
        systemctl restart "$service"

        if systemctl is-active --quiet "$service"; then
            echo "    ✓ $service restarted successfully"
        else
            echo "    ✗ $service failed to restart"
            return 1
        fi
    done

    return 0
}

echo "Secret Rotation"
echo "==============="
echo

# Backup current secrets
cp "$SECRETS_FILE" "$SECRETS_FILE.backup-$(date +%Y%m%d_%H%M%S)"
echo "✓ Backed up secrets"
echo

# Rotate secrets
rotate_secret "API_KEY"
rotate_secret "DB_PASSWORD"
rotate_secret "JWT_SECRET"
rotate_secret "ENCRYPTION_KEY"

# Reload services with new secrets
if reload_services; then
    echo
    echo "✓ Secret rotation complete"
else
    echo
    echo "✗ Failed to reload services"
    echo "Restoring backup..."
    cp "$SECRETS_FILE.backup-"* "$SECRETS_FILE"
    exit 1
fi
