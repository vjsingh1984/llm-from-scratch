#!/bin/bash
# Manage firewall rules

ACTION="${1}"
PORT="${2}"
PROTOCOL="${3:-tcp}"

usage() {
    echo "Usage: $0 <allow|deny|list> [port] [protocol]"
    exit 1
}

list_rules() {
    echo "Current Firewall Rules"
    echo "======================"

    if command -v ufw &>/dev/null; then
        ufw status numbered
    elif command -v firewall-cmd &>/dev/null; then
        firewall-cmd --list-all
    elif command -v iptables &>/dev/null; then
        iptables -L -n -v
    else
        echo "No firewall found"
        exit 1
    fi
}

allow_port() {
    local port=$1
    local proto=$2

    echo "Allowing $proto port $port..."

    if command -v ufw &>/dev/null; then
        ufw allow "$port/$proto"
    elif command -v firewall-cmd &>/dev/null; then
        firewall-cmd --permanent --add-port="$port/$proto"
        firewall-cmd --reload
    elif command -v iptables &>/dev/null; then
        iptables -A INPUT -p "$proto" --dport "$port" -j ACCEPT
    fi

    echo "✓ Port $port/$proto allowed"
}

deny_port() {
    local port=$1
    local proto=$2

    echo "Denying $proto port $port..."

    if command -v ufw &>/dev/null; then
        ufw deny "$port/$proto"
    elif command -v firewall-cmd &>/dev/null; then
        firewall-cmd --permanent --remove-port="$port/$proto"
        firewall-cmd --reload
    elif command -v iptables &>/dev/null; then
        iptables -A INPUT -p "$proto" --dport "$port" -j DROP
    fi

    echo "✓ Port $port/$proto denied"
}

case $ACTION in
    allow)
        [ -z "$PORT" ] && usage
        allow_port "$PORT" "$PROTOCOL"
        ;;
    deny)
        [ -z "$PORT" ] && usage
        deny_port "$PORT" "$PROTOCOL"
        ;;
    list)
        list_rules
        ;;
    *)
        usage
        ;;
esac
