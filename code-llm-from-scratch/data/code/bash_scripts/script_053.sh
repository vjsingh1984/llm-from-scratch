#!/bin/bash
# Query cache stats

mysql -e "SHOW STATUS LIKE 'Qcache%';"
