#!/bin/bash
# VPN status checker

echo "VPN Status"
echo "=========="

ip link show | grep -i tun
