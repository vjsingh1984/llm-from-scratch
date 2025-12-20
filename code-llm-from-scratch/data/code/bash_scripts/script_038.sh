#!/bin/bash
# Service dependency graph generator

OUTPUT_FILE="dependencies.dot"

echo "Generating service dependency graph..."

echo "digraph services {" > "$OUTPUT_FILE"

# Scan services and their dependencies
for service in $(kubectl get svc -o name); do
    dependencies=$(kubectl get svc "$service" -o json | jq -r '.metadata.annotations.dependencies')
    echo "  $service -> $dependencies" >> "$OUTPUT_FILE"
done

echo "}" >> "$OUTPUT_FILE"

dot -Tpng "$OUTPUT_FILE" -o dependencies.png

echo "✓ Graph generated: dependencies.png"
