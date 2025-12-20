#!/bin/bash
# Check system security hardening status

echo "System Security Hardening Check"
echo "==============================="
echo

ISSUES=0

check_item() {
    local description=$1
    local command=$2
    local expected=$3

    echo -n "Checking: $description... "

    result=$(eval "$command" 2>/dev/null)

    if [ "$result" = "$expected" ]; then
        echo "✓ PASS"
        return 0
    else
        echo "✗ FAIL (got: $result, expected: $expected)"
        ((ISSUES++))
        return 1
    fi
}

# Check SSH hardening
echo "=== SSH Configuration ==="
check_item "Root login disabled" "grep -E '^PermitRootLogin' /etc/ssh/sshd_config | awk '{print \$2}'" "no"
check_item "Password auth disabled" "grep -E '^PasswordAuthentication' /etc/ssh/sshd_config | awk '{print \$2}'" "no"
check_item "SSH protocol 2" "grep -E '^Protocol' /etc/ssh/sshd_config | awk '{print \$2}'" "2"
echo

# Check firewall
echo "=== Firewall ==="
if command -v ufw &>/dev/null; then
    check_item "UFW enabled" "ufw status | head -1 | awk '{print \$2}'" "active"
elif command -v firewall-cmd &>/dev/null; then
    check_item "Firewalld running" "systemctl is-active firewalld" "active"
else
    echo "✗ No firewall detected"
    ((ISSUES++))
fi
echo

# Check automatic updates
echo "=== Automatic Updates ==="
if [ -f /etc/apt/apt.conf.d/20auto-upgrades ]; then
    check_item "Auto updates enabled" "grep -c '^APT::Periodic::Update-Package-Lists "1"' /etc/apt/apt.conf.d/20auto-upgrades" "1"
else
    echo "✗ Auto updates not configured"
    ((ISSUES++))
fi
echo

# Check system limits
echo "=== System Limits ==="
check_item "Core dumps disabled" "grep -c '* hard core 0' /etc/security/limits.conf" "1"
echo

# Summary
echo "==============================="
if [ $ISSUES -eq 0 ]; then
    echo "✓ All checks passed"
    exit 0
else
    echo "✗ Found $ISSUES security issue(s)"
    exit 1
fi
