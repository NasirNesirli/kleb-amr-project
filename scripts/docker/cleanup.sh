#!/bin/bash
# Cleanup script for AMR K. pneumoniae prediction pipeline Docker resources

set -e

# Default values
IMAGE_NAME="amr-pipeline"
FORCE=false
CLEANUP_VOLUMES=false
CLEANUP_NETWORKS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --volumes)
            CLEANUP_VOLUMES=true
            shift
            ;;
        --networks)
            CLEANUP_NETWORKS=true
            shift
            ;;
        --all)
            CLEANUP_VOLUMES=true
            CLEANUP_NETWORKS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -i, --image NAME    Image name pattern to clean (default: amr-pipeline)"
            echo "  -f, --force         Force removal without confirmation"
            echo "  --volumes          Also remove named volumes"
            echo "  --networks         Also remove networks"
            echo "  --all              Clean volumes and networks"
            echo "  -h, --help         Show this help"
            echo ""
            echo "This script will clean up:"
            echo "  - Stopped containers related to AMR pipeline"
            echo "  - Docker images matching the pattern"
            echo "  - Optionally: volumes and networks"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to ask for confirmation
confirm() {
    if [ "$FORCE" = true ]; then
        return 0
    fi
    
    local message="$1"
    echo -n "$message (y/N): "
    read -r response
    case "$response" in
        [yY]|[yY][eE][sS])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible"
    exit 1
fi

echo "AMR Pipeline Docker Cleanup"
echo "=========================="

# Clean up stopped containers
echo "Checking for stopped containers..."
STOPPED_CONTAINERS=$(docker ps -a --filter "name=amr-" --format "{{.ID}} {{.Names}}" | grep -E "(amr-pipeline|amr-prediction)" || true)

if [ -n "$STOPPED_CONTAINERS" ]; then
    echo "Found stopped containers:"
    echo "$STOPPED_CONTAINERS"
    if confirm "Remove stopped containers?"; then
        echo "$STOPPED_CONTAINERS" | awk '{print $1}' | xargs docker rm
        echo "✓ Stopped containers removed"
    fi
else
    echo "No stopped containers found"
fi

# Clean up images
echo ""
echo "Checking for Docker images..."
IMAGES=$(docker images --filter "reference=$IMAGE_NAME*" --format "{{.ID}} {{.Repository}}:{{.Tag}}" || true)

if [ -n "$IMAGES" ]; then
    echo "Found images:"
    echo "$IMAGES"
    if confirm "Remove Docker images?"; then
        echo "$IMAGES" | awk '{print $1}' | sort -u | xargs docker rmi
        echo "✓ Images removed"
    fi
else
    echo "No matching images found"
fi

# Clean up build cache
echo ""
echo "Checking Docker build cache..."
if confirm "Remove Docker build cache?"; then
    docker builder prune -f
    echo "✓ Build cache cleared"
fi

# Clean up volumes
if [ "$CLEANUP_VOLUMES" = true ]; then
    echo ""
    echo "Checking for named volumes..."
    VOLUMES=$(docker volume ls --filter "name=.*conda-cache.*|.*snakemake-cache.*" --format "{{.Name}}" || true)
    
    if [ -n "$VOLUMES" ]; then
        echo "Found volumes:"
        echo "$VOLUMES"
        if confirm "Remove named volumes? (This will delete cached environments)"; then
            echo "$VOLUMES" | xargs docker volume rm
            echo "✓ Volumes removed"
        fi
    else
        echo "No matching volumes found"
    fi
fi

# Clean up networks
if [ "$CLEANUP_NETWORKS" = true ]; then
    echo ""
    echo "Checking for custom networks..."
    NETWORKS=$(docker network ls --filter "name=amr-pipeline-network" --format "{{.ID}} {{.Name}}" || true)
    
    if [ -n "$NETWORKS" ]; then
        echo "Found networks:"
        echo "$NETWORKS"
        if confirm "Remove custom networks?"; then
            echo "$NETWORKS" | awk '{print $1}' | xargs docker network rm
            echo "✓ Networks removed"
        fi
    else
        echo "No matching networks found"
    fi
fi

# General Docker cleanup
echo ""
if confirm "Run general Docker system cleanup?"; then
    docker system prune -f
    echo "✓ General cleanup completed"
fi

echo ""
echo "Cleanup completed!"
echo ""
echo "Remaining Docker resources:"
echo "Images:"
docker images | grep -E "(amr-pipeline|REPOSITORY)" || echo "  No AMR pipeline images found"
echo ""
echo "Containers:"
docker ps -a | grep -E "(amr-|CONTAINER)" || echo "  No AMR pipeline containers found"

if [ "$CLEANUP_VOLUMES" = true ]; then
    echo ""
    echo "Volumes:"
    docker volume ls | grep -E "(conda-cache|snakemake-cache|NAME)" || echo "  No cache volumes found"
fi