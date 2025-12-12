#!/bin/bash
# Deploy AMR pipeline to Google Cloud Run

set -e

# Default values
PROJECT_ID=""
REGION="us-central1"
SERVICE_NAME="amr-pipeline"
MEMORY="16Gi"
CPU="4"
TIMEOUT="3600"
MAX_INSTANCES="5"
IMAGE_TAG="latest"
ALLOW_UNAUTHENTICATED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project)
            PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -s|--service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -c|--cpu)
            CPU="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --allow-unauthenticated)
            ALLOW_UNAUTHENTICATED=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -p, --project ID      GCP Project ID (required)"
            echo "  -r, --region REGION   GCP region (default: us-central1)"
            echo "  -s, --service NAME    Cloud Run service name (default: amr-pipeline)"
            echo "  -m, --memory SIZE     Memory allocation (default: 16Gi)"
            echo "  -c, --cpu COUNT       CPU allocation (default: 4)"
            echo "  -t, --timeout SEC     Request timeout (default: 3600)"
            echo "  --max-instances NUM   Maximum instances (default: 5)"
            echo "  --tag TAG            Image tag (default: latest)"
            echo "  --allow-unauthenticated  Allow unauthenticated access"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --project my-project --allow-unauthenticated"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$PROJECT_ID" ]; then
    echo "Error: Project ID is required. Use --project or set PROJECT_ID environment variable."
    exit 1
fi

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Error: Not authenticated with gcloud. Run 'gcloud auth login'"
    exit 1
fi

echo "Deploying AMR pipeline to Google Cloud Run"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Memory: $MEMORY"
echo "CPU: $CPU"
echo "Image tag: $IMAGE_TAG"
echo ""

# Set the project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --quiet

# Create Artifact Registry repository if it doesn't exist
REPO_NAME="amr-pipeline"
echo "Setting up Artifact Registry..."
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" &>/dev/null; then
    echo "Creating Artifact Registry repository..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="AMR K. pneumoniae prediction pipeline"
else
    echo "Artifact Registry repository already exists"
fi

# Configure Docker authentication
echo "Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build and push the image
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:${IMAGE_TAG}"
echo "Building and pushing image to $IMAGE_URL..."

# Get script directory to ensure we're in the right location for building
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Build using Cloud Build for better performance
gcloud builds submit \
    --tag "$IMAGE_URL" \
    --machine-type=e2-highcpu-8 \
    --disk-size=100 \
    --quiet

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
DEPLOY_ARGS=(
    "run" "deploy" "$SERVICE_NAME"
    "--image" "$IMAGE_URL"
    "--platform" "managed"
    "--region" "$REGION"
    "--memory" "$MEMORY"
    "--cpu" "$CPU"
    "--timeout" "$TIMEOUT"
    "--max-instances" "$MAX_INSTANCES"
    "--set-env-vars" "SNAKEMAKE_CORES=$CPU,SNAKEMAKE_MEMORY=${MEMORY%Gi}G"
    "--execution-environment" "gen2"
    "--quiet"
)

if [ "$ALLOW_UNAUTHENTICATED" = true ]; then
    DEPLOY_ARGS+=("--allow-unauthenticated")
fi

gcloud "${DEPLOY_ARGS[@]}"

# Get the service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --format="value(status.url)")

echo ""
echo "✅ Deployment completed successfully!"
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test the deployment:"
echo "curl \"$SERVICE_URL\" -H \"Content-Type: application/json\" -d '{\"target\": \"--help\"}'"
echo ""
echo "View logs:"
echo "gcloud logs read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --limit=50"
echo ""
echo "Update the service:"
echo "gcloud run services update $SERVICE_NAME --region=$REGION"