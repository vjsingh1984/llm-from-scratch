#!/bin/bash
# SSL certificate expiration checker

DOMAINS=("example.com" "api.example.com" "www.example.com")
WARNING_DAYS=30
ALERT_EMAIL="admin@example.com"

check_cert() {
    local domain=$1

    echo "Checking certificate for $domain..."

    # Get expiration date
    expiry=$(echo | openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null |              openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)

    if [ -z "$expiry" ]; then
        echo "ERROR: Could not retrieve certificate for $domain"
        return 1
    fi

    # Calculate days until expiration
    expiry_epoch=$(date -d "$expiry" +%s 2>/dev/null || date -j -f "%b %d %H:%M:%S %Y %Z" "$expiry" +%s)
    current_epoch=$(date +%s)
    days_left=$(( ($expiry_epoch - $current_epoch) / 86400 ))

    echo "  Expires: $expiry"
    echo "  Days left: $days_left"

    if [ $days_left -lt 0 ]; then
        echo "  STATUS: EXPIRED"
        echo "$domain certificate has EXPIRED!" | mail -s "SSL ALERT: Certificate Expired" "$ALERT_EMAIL"
        return 1
    elif [ $days_left -lt $WARNING_DAYS ]; then
        echo "  STATUS: WARNING"
        echo "$domain certificate expires in $days_left days" | mail -s "SSL WARNING: Certificate Expiring Soon" "$ALERT_EMAIL"
        return 1
    else
        echo "  STATUS: OK"
        return 0
    fi
}

echo "SSL Certificate Check"
echo "====================="
echo

ISSUES=0

for domain in "${DOMAINS[@]}"; do
    check_cert "$domain" || ((ISSUES++))
    echo
done

if [ $ISSUES -gt 0 ]; then
    echo "Certificate issues detected: $ISSUES"
    exit 1
else
    echo "All certificates OK"
    exit 0
fi
