#!/bin/bash
# Log shipping to central server

REMOTE_HOST="log-server.example.com"

rsync -az /var/log/ "$REMOTE_HOST:/logs/$(hostname)/"
