#!/bin/bash

# Script to build and run the Knowledge Access Analytics Docker container

set -e

echo "🚀 Building Knowledge Access Analytics Docker image..."

# Build the Docker image using Red Hat UBI
docker build -t knowledge-access-analytics:latest .

echo "✅ Build completed successfully!"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "   Please copy .env.example to .env and add your API keys"
    echo "   cp .env.example .env"
    echo ""
fi

echo "🏃 Running Docker container..."

# Run the container with environment file and port mapping
docker run -d \
    --name knowledge-access-analytics \
    -p 8080:8080 \
    --env-file .env \
    knowledge-access-analytics:latest

echo "✅ Container started successfully!"
echo ""
echo "📊 Container status:"
docker ps -f name=knowledge-access-analytics

echo ""
echo "🌐 Application is running at:"
echo "   Local: http://localhost:8080"
echo "   Network: http://0.0.0.0:8080"
echo ""
echo "📝 To view logs:"
echo "   docker logs -f knowledge-access-analytics"
echo ""
echo "🛑 To stop the container:"
echo "   docker stop knowledge-access-analytics"
echo "   docker rm knowledge-access-analytics"