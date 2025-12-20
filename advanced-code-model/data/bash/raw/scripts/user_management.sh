#!/bin/bash
# User management automation

function create_user() {
    local username=$1
    local fullname=$2
    
    if id "$username" &>/dev/null; then
        echo "User $username already exists"
        return 1
    fi
    
    useradd -m -s /bin/bash -c "$fullname" "$username"
    passwd "$username"
    echo "User $username created successfully"
}

function delete_user() {
    local username=$1
    
    if ! id "$username" &>/dev/null; then
        echo "User $username does not exist"
        return 1
    fi
    
    userdel -r "$username"
    echo "User $username deleted"
}

case "$1" in
    add)
        create_user "$2" "$3"
        ;;
    remove)
        delete_user "$2"
        ;;
    *)
        echo "Usage: $0 {add|remove} username [fullname]"
        exit 1
        ;;
esac
