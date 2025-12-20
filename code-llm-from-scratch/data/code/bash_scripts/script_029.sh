#!/bin/bash
# Publish build artifacts to repository

ARTIFACT_FILE="${1}"
REPO_URL="${2:-https://artifacts.example.com}"
VERSION="${3:-$(git describe --tags --always)}"

if [ -z "$ARTIFACT_FILE" ] || [ ! -f "$ARTIFACT_FILE" ]; then
    echo "Usage: $0 <artifact-file> [repo-url] [version]"
    exit 1
fi

echo "Publishing Artifact"
echo "==================="
echo "File: $ARTIFACT_FILE"
echo "Version: $VERSION"
echo "Repository: $REPO_URL"
echo

# Calculate checksum
echo "Calculating checksums..."
SHA256=$(sha256sum "$ARTIFACT_FILE" | awk '{print $1}')
MD5=$(md5sum "$ARTIFACT_FILE" | awk '{print $1}')

echo "  SHA256: $SHA256"
echo "  MD5: $MD5"

# Create metadata
METADATA_FILE="${ARTIFACT_FILE}.metadata.json"
cat > "$METADATA_FILE" <<EOF
{
  "filename": "$(basename "$ARTIFACT_FILE")",
  "version": "$VERSION",
  "size": $(stat -f%z "$ARTIFACT_FILE" 2>/dev/null || stat -c%s "$ARTIFACT_FILE"),
  "sha256": "$SHA256",
  "md5": "$MD5",
  "build_date": "$(date -Iseconds)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
}
EOF

echo
echo "✓ Metadata created: $METADATA_FILE"

# Upload artifact
echo
echo "Uploading to repository..."

curl -f -X POST     -H "Content-Type: application/octet-stream"     -H "X-Artifact-Version: $VERSION"     --data-binary "@$ARTIFACT_FILE"     "$REPO_URL/upload"

if [ $? -eq 0 ]; then
    echo "✓ Artifact published successfully"

    # Upload metadata
    curl -f -X POST         -H "Content-Type: application/json"         --data-binary "@$METADATA_FILE"         "$REPO_URL/metadata"

    echo "✓ Metadata published"
else
    echo "✗ Failed to publish artifact"
    exit 1
fi
