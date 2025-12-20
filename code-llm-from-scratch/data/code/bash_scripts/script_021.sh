#!/bin/bash
# Docker system cleanup script

echo "Docker System Cleanup"
echo "===================="

# Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove dangling images
echo "Removing dangling images..."
docker image prune -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# Show disk usage
echo
echo "Current disk usage:"
docker system df

echo
echo "Cleanup complete"
