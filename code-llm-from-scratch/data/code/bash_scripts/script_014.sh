#!/bin/bash
# Audit file permissions for security issues

SEARCH_DIR="${1:-/}"
REPORT_FILE="/tmp/permission-audit-$(date +%Y%m%d).txt"

{
    echo "File Permission Security Audit"
    echo "=============================="
    echo "Date: $(date)"
    echo "Search directory: $SEARCH_DIR"
    echo

    echo "=== World-Writable Files ==="
    find "$SEARCH_DIR" -type f -perm -002 ! -path "*/proc/*" ! -path "*/sys/*" 2>/dev/null

    echo
    echo "=== World-Writable Directories ==="
    find "$SEARCH_DIR" -type d -perm -002 ! -path "*/proc/*" ! -path "*/sys/*" 2>/dev/null

    echo
    echo "=== SUID Files ==="
    find "$SEARCH_DIR" -type f -perm -4000 ! -path "*/proc/*" 2>/dev/null

    echo
    echo "=== SGID Files ==="
    find "$SEARCH_DIR" -type f -perm -2000 ! -path "*/proc/*" 2>/dev/null

    echo
    echo "=== Files Without Owner ==="
    find "$SEARCH_DIR" -nouser ! -path "*/proc/*" ! -path "*/sys/*" 2>/dev/null

    echo
    echo "=== Files Without Group ==="
    find "$SEARCH_DIR" -nogroup ! -path "*/proc/*" ! -path "*/sys/*" 2>/dev/null

} | tee "$REPORT_FILE"

echo
echo "Audit report saved to: $REPORT_FILE"
