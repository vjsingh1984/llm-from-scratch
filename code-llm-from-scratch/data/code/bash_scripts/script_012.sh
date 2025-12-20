#!/bin/bash
# Find and optionally remove broken symbolic links

SEARCH_DIR="${1:-.}"
DELETE=false

usage() {
    echo "Usage: $0 [directory] [-d]"
    echo "  -d: Delete broken symlinks"
    exit 1
}

while getopts "d" opt; do
    case $opt in
        d) DELETE=true ;;
        *) usage ;;
    esac
done

echo "Searching for broken symlinks in: $SEARCH_DIR"
echo "Delete mode: $DELETE"
echo

BROKEN_LINKS=()

# Find all symlinks
while IFS= read -r link; do
    # Check if target exists
    if [ ! -e "$link" ]; then
        BROKEN_LINKS+=("$link")
        echo "Broken: $link -> $(readlink "$link")"

        if [ "$DELETE" = true ]; then
            rm "$link"
            echo "  Deleted"
        fi
    fi
done < <(find "$SEARCH_DIR" -type l 2>/dev/null)

echo
echo "Found ${#BROKEN_LINKS[@]} broken symlink(s)"

if [ "$DELETE" = false ] && [ ${#BROKEN_LINKS[@]} -gt 0 ]; then
    echo "Run with -d to delete broken symlinks"
fi
