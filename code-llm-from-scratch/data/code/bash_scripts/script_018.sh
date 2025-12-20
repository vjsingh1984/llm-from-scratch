#!/bin/bash
# Check available system patches and security updates

echo "System Patch Status"
echo "==================="
echo "Date: $(date)"
echo

# Detect package manager
if command -v apt-get &>/dev/null; then
    PKG_MGR="apt"
elif command -v yum &>/dev/null; then
    PKG_MGR="yum"
else
    echo "Error: Unsupported package manager"
    exit 1
fi

update_cache() {
    echo "Updating package cache..."
    if [ "$PKG_MGR" = "apt" ]; then
        apt-get update -qq
    else
        yum check-update -q
    fi
}

check_updates() {
    echo
    echo "=== Available Updates ==="

    if [ "$PKG_MGR" = "apt" ]; then
        TOTAL=$(apt list --upgradable 2>/dev/null | grep -c upgradable)
        SECURITY=$(apt list --upgradable 2>/dev/null | grep -i security | wc -l)

        echo "Total updates available: $TOTAL"
        echo "Security updates: $SECURITY"
        echo

        if [ $SECURITY -gt 0 ]; then
            echo "Security Updates:"
            apt list --upgradable 2>/dev/null | grep -i security
        fi
    else
        TOTAL=$(yum list updates 2>/dev/null | grep -v "^Loaded" | grep -v "^Updated" | wc -l)
        SECURITY=$(yum list-security 2>/dev/null | grep -i security | wc -l)

        echo "Total updates available: $TOTAL"
        echo "Security updates: $SECURITY"
    fi
}

check_reboot_required() {
    echo
    echo "=== Reboot Status ==="

    if [ -f /var/run/reboot-required ]; then
        echo "⚠ System reboot required"
        if [ -f /var/run/reboot-required.pkgs ]; then
            echo "Packages requiring reboot:"
            cat /var/run/reboot-required.pkgs
        fi
    else
        echo "✓ No reboot required"
    fi
}

last_update() {
    echo
    echo "=== Last Update ==="

    if [ "$PKG_MGR" = "apt" ]; then
        LAST=$(stat -c %y /var/lib/apt/periodic/update-success-stamp 2>/dev/null || echo "Unknown")
    else
        LAST=$(stat -c %y /var/cache/yum 2>/dev/null || echo "Unknown")
    fi

    echo "Last update check: $LAST"
}

update_cache
check_updates
check_reboot_required
last_update

echo
echo "==================="
echo "Patch check complete"
