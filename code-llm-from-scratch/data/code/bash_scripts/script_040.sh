#!/bin/bash
# Infrastructure drift checker

TERRAFORM_DIR="${1:-.}"

cd "$TERRAFORM_DIR"

echo "Checking infrastructure drift..."

terraform plan -detailed-exitcode

case $? in
    0)
        echo "✓ No drift detected"
        ;;
    2)
        echo "⚠ Drift detected"
        terraform plan
        ;;
    *)
        echo "✗ Error running terraform"
        exit 1
        ;;
esac
