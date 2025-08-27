#!/bin/bash
# Build script for the rainfall prediction API

set -e

echo "Building rainfall prediction API Docker image..."

# Move to repo root and build the Docker image
cd ..
docker build -t rainfall-inference:latest .

echo "✓ Docker image built successfully!"
echo ""
echo "To run the container:"
echo "  docker run --rm -p 8080:8080 rainfall-inference:latest"
echo ""
echo "To test the API:"
echo "  curl http://localhost:8080/healthz"
echo "  curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d @api/test_request.json"
