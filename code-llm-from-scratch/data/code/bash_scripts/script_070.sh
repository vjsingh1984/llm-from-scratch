#!/bin/bash
# Network scan for live hosts

NETWORK="${1:-192.168.1.0/24}"

echo "Scanning network: $NETWORK"

nmap -sn "$NETWORK" | grep "Nmap scan report"
