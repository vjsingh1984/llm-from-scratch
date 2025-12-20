#!/bin/bash
# Check service dependencies and start order

SERVICE="${1:-nginx}"

if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service-name>"
    exit 1
fi

echo "Service Dependency Analysis: $SERVICE"
echo "====================================="
echo

# Get service file
SERVICE_FILE=$(systemctl show -p FragmentPath "$SERVICE" | cut -d= -f2)

if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file not found for $SERVICE"
    exit 1
fi

echo "Service file: $SERVICE_FILE"
echo

# Parse dependencies
echo "=== Dependencies ==="
echo "After:"
grep "^After=" "$SERVICE_FILE" | cut -d= -f2 | tr ' ' '
' | sed 's/^/  - /'

echo
echo "Requires:"
grep "^Requires=" "$SERVICE_FILE" | cut -d= -f2 | tr ' ' '
' | sed 's/^/  - /'

echo
echo "Wants:"
grep "^Wants=" "$SERVICE_FILE" | cut -d= -f2 | tr ' ' '
' | sed 's/^/  - /'

echo
echo "=== Dependents (services that depend on this) ==="
for unit in /etc/systemd/system/*.service /lib/systemd/system/*.service; do
    if [ -f "$unit" ]; then
        if grep -q "$SERVICE" "$unit" 2>/dev/null; then
            echo "  - $(basename "$unit")"
        fi
    fi
done

echo
echo "=== Current Status ==="
systemctl status "$SERVICE" --no-pager
