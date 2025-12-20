#!/bin/bash
# Custom metric exporter for Prometheus

echo "# HELP system_cpu_usage CPU usage percentage"
echo "# TYPE system_cpu_usage gauge"
echo "system_cpu_usage $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)"
