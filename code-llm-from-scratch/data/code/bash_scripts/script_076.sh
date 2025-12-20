#!/bin/bash
# Alert manager

MESSAGE="${1}"
SEVERITY="${2:-INFO}"

echo "[$SEVERITY] $MESSAGE" | tee -a /var/log/alerts.log

case $SEVERITY in
    CRITICAL|ERROR)
        echo "$MESSAGE" | mail -s "[$SEVERITY] Alert" admin@example.com
        ;;
esac
