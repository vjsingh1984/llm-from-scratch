#!/bin/bash
# Log rotation status

echo "Log Rotation Status"
echo "==================="

logrotate -d /etc/logrotate.conf 2>&1 | grep -E "(rotating|compressing)"
