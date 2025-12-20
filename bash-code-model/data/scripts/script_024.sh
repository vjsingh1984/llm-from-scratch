#!/bin/bash
declare -A config
config[host]="localhost"
config[port]="8080"
echo "${config[host]}:${config[port]}"
