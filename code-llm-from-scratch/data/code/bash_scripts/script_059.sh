#!/bin/bash
# Advanced DNS lookup tool

DOMAIN="${1}"

if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain>"
    exit 1
fi

echo "DNS Lookup: $DOMAIN"
echo "==================="
echo

echo "=== A Records ==="
dig +short A "$DOMAIN"

echo
echo "=== AAAA Records (IPv6) ==="
dig +short AAAA "$DOMAIN"

echo
echo "=== MX Records ==="
dig +short MX "$DOMAIN"

echo
echo "=== NS Records ==="
dig +short NS "$DOMAIN"

echo
echo "=== TXT Records ==="
dig +short TXT "$DOMAIN"

echo
echo "=== SOA Record ==="
dig +short SOA "$DOMAIN"

echo
echo "=== Reverse DNS ==="
IP=$(dig +short A "$DOMAIN" | head -1)
if [ -n "$IP" ]; then
    dig +short -x "$IP"
fi
