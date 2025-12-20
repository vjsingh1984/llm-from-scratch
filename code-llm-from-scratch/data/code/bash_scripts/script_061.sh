#!/bin/bash
# SSL certificate checker

DOMAIN="${1}"
PORT="${2:-443}"

echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:$PORT" 2>/dev/null |     openssl x509 -noout -dates -subject -issuer
