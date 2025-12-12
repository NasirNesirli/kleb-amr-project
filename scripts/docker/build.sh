#!/bin/bash
# Build script for AMR K. pneumoniae prediction pipeline Docker container

set -e

# Default values
IMAGE_NAME="amr-pipeline"
IMAGE_TAG="latest"
BUILD_TARGET="production"
NO_CACHE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -n, --name NAME       Docker image name (default: amr-pipeline)"
            echo "  -t, --tag TAG         Docker image tag (default: latest)"
            echo "  --target TARGET       Build target (default: production)"
            echo "  --no-cache           Build without using cache"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible"
    exit 1
fi

# Get script directory to ensure we're in the right location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Building AMR pipeline Docker container..."
echo "Project root: $PROJECT_ROOT"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Target: $BUILD_TARGET"

# Change to project root
cd "$PROJECT_ROOT"

# Build arguments
BUILD_ARGS=(
    "--build-arg" "BUILDKIT_INLINE_CACHE=1"
    "--target" "$BUILD_TARGET"
    "--tag" "$IMAGE_NAME:$IMAGE_TAG"
)

if [ "$NO_CACHE" = true ]; then
    BUILD_ARGS+=("--no-cache")
fi

# Add latest tag for production builds
if [ "$BUILD_TARGET" = "production" ] && [ "$IMAGE_TAG" != "latest" ]; then
    BUILD_ARGS+=("--tag" "$IMAGE_NAME:latest")
fi

# Build the container
echo "Running: docker build ${BUILD_ARGS[*]} ."
docker build "${BUILD_ARGS[@]}" .

# Verify the build
echo ""
echo "Build completed successfully!"
echo "Image size:"
docker images "$IMAGE_NAME:$IMAGE_TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Test basic functionality
echo ""
echo "Testing container..."
if docker run --rm "$IMAGE_NAME:$IMAGE_TAG" --version >/dev/null 2>&1; then
    echo "✓ Container test passed - Snakemake is working"
else
    echo "✗ Container test failed - Snakemake not working properly"
    exit 1
fi

echo ""
echo "Container built and tested successfully!"
echo "To run the pipeline:"
echo "  docker run -v \$(pwd)/data:/pipeline/data -v \$(pwd)/results:/pipeline/results $IMAGE_NAME:$IMAGE_TAG --cores 4"
echo ""
echo "To use docker-compose:"
echo "  docker-compose up amr-pipeline"