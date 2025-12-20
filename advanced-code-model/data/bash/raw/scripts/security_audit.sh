#!/bin/bash
# Basic security audit

echo "Security Audit Report"
echo "===================="
echo "Generated: $(date)"
echo ""

# Check for users with empty passwords
echo "Users with empty passwords:"
awk -F: '($2 == "") {print $1}' /etc/shadow 2>/dev/null || echo "  None (or insufficient permissions)"
echo ""

# Check for UID 0 users
echo "Users with UID 0 (besides root):"
awk -F: '($3 == 0 && $1 != "root") {print $1}' /etc/passwd || echo "  None"
echo ""

# Check SSH configuration
echo "SSH Security Settings:"
if [ -f /etc/ssh/sshd_config ]; then
    echo "  PermitRootLogin: $(grep '^PermitRootLogin' /etc/ssh/sshd_config | awk '{print $2}')"
    echo "  PasswordAuthentication: $(grep '^PasswordAuthentication' /etc/ssh/sshd_config | awk '{print $2}')"
fi
echo ""

# Check firewall status
echo "Firewall Status:"
if command -v ufw >/dev/null; then
    ufw status
elif command -v firewall-cmd >/dev/null; then
    firewall-cmd --state
else
    echo "  No firewall detected"
fi
