#!/bin/bash

# Build script for Knowledge Access Analytics Docker image

set -e

echo "Building Knowledge Access Analytics Docker image..."

# Build the Docker image using Red Hat UBI
docker build -t knowledge-access-analytics:latest .

echo "Build completed successfully!"
echo ""
echo "To run locally:"
echo "docker run -p 8080:8080 --env-file .env knowledge-access-analytics:latest"
echo ""
echo "To push to registry:"
echo "docker tag knowledge-access-analytics:latest your-registry/knowledge-access-analytics:latest"
echo "docker push your-registry/knowledge-access-analytics:latest"