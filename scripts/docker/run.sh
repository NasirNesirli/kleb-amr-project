#!/bin/bash
# Run script for AMR K. pneumoniae prediction pipeline Docker container

set -e

# Default values
IMAGE_NAME="amr-pipeline"
IMAGE_TAG="latest"
CORES=4
MEMORY="8G"
DRY_RUN=false
INTERACTIVE=false
TARGET=""
DATA_DIR="$(pwd)/data"
RESULTS_DIR="$(pwd)/results"
CONFIG_DIR="$(pwd)/config"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -c|--cores)
            CORES="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options] [snakemake arguments]"
            echo ""
            echo "Options:"
            echo "  -i, --image NAME      Docker image name (default: amr-pipeline)"
            echo "  -t, --tag TAG         Docker image tag (default: latest)"
            echo "  -c, --cores NUM       Number of cores to use (default: 4)"
            echo "  -m, --memory SIZE     Memory limit (default: 8G)"
            echo "  --data-dir PATH       Data directory path (default: ./data)"
            echo "  --results-dir PATH    Results directory path (default: ./results)"
            echo "  --config-dir PATH     Config directory path (default: ./config)"
            echo "  --dry-run            Run pipeline in dry-run mode"
            echo "  --interactive        Run container interactively"
            echo "  --target TARGET      Run specific Snakemake target"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run                    # Check pipeline without running"
            echo "  $0 --cores 8                    # Run with 8 cores"
            echo "  $0 --target tree_models         # Run only tree models"
            echo "  $0 --interactive                # Interactive shell in container"
            echo "  $0 results/models/xgboost/amikacin_results.json  # Run specific rule"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            # Remaining arguments are passed to Snakemake
            break
            ;;
    esac
done

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible"
    exit 1
fi

# Check if image exists
if ! docker image inspect "$IMAGE_NAME:$IMAGE_TAG" >/dev/null 2>&1; then
    echo "Error: Docker image $IMAGE_NAME:$IMAGE_TAG not found"
    echo "Please build the image first using: ./scripts/docker/build.sh"
    exit 1
fi

# Ensure directories exist
mkdir -p "$RESULTS_DIR" "$DATA_DIR"

# Check if data directory has content
if [ ! -f "$DATA_DIR/metadata.csv" ]; then
    echo "Warning: $DATA_DIR/metadata.csv not found"
    echo "Make sure your data directory contains the required input files"
fi

# Build Docker run command
DOCKER_ARGS=(
    "--rm"
    "--volume" "$DATA_DIR:/pipeline/data:ro"
    "--volume" "$RESULTS_DIR:/pipeline/results:rw"
    "--volume" "$CONFIG_DIR:/pipeline/config:ro"
    "--memory" "$MEMORY"
    "--cpus" "$CORES"
    "--name" "amr-pipeline-run-$$"
)

# Add interactive flags if requested
if [ "$INTERACTIVE" = true ]; then
    DOCKER_ARGS+=("-it" "--entrypoint" "/bin/bash")
    SNAKEMAKE_ARGS=()
else
    # Build Snakemake arguments
    SNAKEMAKE_ARGS=(
        "--use-conda"
        "--conda-frontend" "mamba"
        "--cores" "$CORES"
        "--rerun-incomplete"
        "--printshellcmds"
    )
    
    if [ "$DRY_RUN" = true ]; then
        SNAKEMAKE_ARGS+=("--dry-run")
    fi
    
    if [ -n "$TARGET" ]; then
        SNAKEMAKE_ARGS+=("$TARGET")
    fi
    
    # Add any additional arguments passed to the script
    SNAKEMAKE_ARGS+=("$@")
fi

echo "Running AMR pipeline container..."
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Cores: $CORES"
echo "Memory: $MEMORY"
echo "Data directory: $DATA_DIR"
echo "Results directory: $RESULTS_DIR"

if [ "$INTERACTIVE" = true ]; then
    echo "Starting interactive shell..."
    docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME:$IMAGE_TAG"
else
    echo "Snakemake arguments: ${SNAKEMAKE_ARGS[*]}"
    echo ""
    
    # Run the container
    docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME:$IMAGE_TAG" "${SNAKEMAKE_ARGS[@]}"
    
    echo ""
    echo "Pipeline execution completed!"
    echo "Results available in: $RESULTS_DIR"
fi