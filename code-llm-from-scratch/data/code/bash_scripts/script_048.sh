#!/bin/bash
# Database user audit

echo "Database User Audit"
echo "==================="

mysql -e "
    SELECT
        user,
        host,
        password_expired,
        password_last_changed,
        password_lifetime
    FROM mysql.user
    ORDER BY user, host;
"
