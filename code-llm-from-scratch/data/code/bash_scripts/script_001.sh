#!/bin/bash
# Batch user creation from CSV file
# Usage: ./create_users.sh users.csv

CSV_FILE="${1:-users.csv}"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

while IFS=, read -r username fullname email; do
    # Skip header
    if [ "$username" = "username" ]; then
        continue
    fi

    echo "Creating user: $username"

    # Create user
    useradd -m -c "$fullname" "$username"

    # Set default password (force change on first login)
    echo "$username:Change@123" | chpasswd
    chage -d 0 "$username"

    echo "User $username created successfully"
done < "$CSV_FILE"

echo "Batch user creation complete"
