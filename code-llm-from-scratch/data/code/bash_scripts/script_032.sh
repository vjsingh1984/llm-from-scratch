#!/bin/bash
# Service mesh configuration

SERVICE="${1}"
MESH_CONFIG="/etc/istio/configs/${SERVICE}.yaml"

echo "Configuring service mesh for: $SERVICE"

# Generate mesh config
cat > "$MESH_CONFIG" <<EOF
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: $SERVICE
spec:
  hosts:
  - $SERVICE
  http:
  - route:
    - destination:
        host: $SERVICE
EOF

echo "✓ Mesh configured"
