#!/bin/bash
# Docker resource cleanup

echo "Docker Cleanup Script"
echo "===================="

# Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused images
echo "Removing unused images..."
docker image prune -a -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# Show disk usage
echo ""
echo "Current Docker disk usage:"
docker system df

echo ""
echo "Cleanup complete!"
