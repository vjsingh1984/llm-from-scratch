#!/bin/bash
# Batch git repository updater

REPO_DIR="${1:-.}"

echo "Updating all git repositories in $REPO_DIR..."

find "$REPO_DIR" -name ".git" -type d | while read gitdir; do
    repo=$(dirname "$gitdir")
    echo
    echo "=== $(basename "$repo") ==="
    cd "$repo"

    # Check for changes
    if [[ -n $(git status -s) ]]; then
        echo "  ⚠ Uncommitted changes"
        git status -s
    else
        # Pull updates
        echo "  Pulling updates..."
        git pull --rebase

        if [ $? -eq 0 ]; then
            echo "  ✓ Updated successfully"
        else
            echo "  ✗ Update failed"
        fi
    fi
done

echo
echo "All repositories processed"
