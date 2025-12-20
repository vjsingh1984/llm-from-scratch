#!/bin/bash
# Asset compilation and deployment

echo "Compiling assets..."

npm run build
aws s3 sync dist/ s3://assets-bucket/

echo "✓ Assets deployed"
