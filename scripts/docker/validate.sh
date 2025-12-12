#!/bin/bash
# Validation script for Docker setup

set -e

echo "Validating Docker setup for AMR pipeline..."
echo "==========================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker daemon is not running"
    echo "Please start Docker and try again"
    exit 1
fi

echo "✅ Docker is available and running"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Validate Dockerfile syntax by attempting to parse it
echo ""
echo "Validating Dockerfile syntax..."
if docker build --help >/dev/null 2>&1; then
    # Try a basic syntax check by building only the first stage
    if docker build --target base --no-cache -t test-syntax . >/dev/null 2>&1; then
        echo "✅ Dockerfile syntax appears valid"
        docker rmi test-syntax >/dev/null 2>&1 || true
    else
        echo "❌ Dockerfile build failed - checking syntax..."
        docker build --target base --no-cache -t test-syntax . 2>&1 | head -20
        exit 1
    fi
else
    echo "⚠️  Cannot validate Dockerfile syntax"
fi

# Validate docker-compose.yml
echo ""
echo "Validating docker-compose.yml..."
if command -v docker-compose &> /dev/null; then
    if docker-compose config >/dev/null 2>&1; then
        echo "✅ docker-compose.yml is valid"
    else
        echo "❌ docker-compose.yml syntax error"
        docker-compose config
        exit 1
    fi
else
    echo "⚠️  docker-compose not installed - skipping validation"
fi

# Check required files
echo ""
echo "Checking required files..."
REQUIRED_FILES=(
    "Dockerfile"
    ".dockerignore"
    "docker-compose.yml"
    "Makefile"
    "Snakefile"
    "config/config.yaml"
    "scripts/docker/build.sh"
    "scripts/docker/run.sh"
    "scripts/docker/cleanup.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Check script permissions
echo ""
echo "Checking script permissions..."
SCRIPTS=(
    "scripts/docker/build.sh"
    "scripts/docker/run.sh"
    "scripts/docker/cleanup.sh"
    "scripts/gcp/deploy-cloudrun.sh"
    "scripts/gcp/setup-gke.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "✅ $script (executable)"
        else
            echo "⚠️  $script (not executable, fixing...)"
            chmod +x "$script"
            echo "✅ $script (fixed)"
        fi
    fi
done

# Validate environment files
echo ""
echo "Checking conda environment files..."
ENV_COUNT=$(find envs/ -name "*.yaml" 2>/dev/null | wc -l || echo "0")
if [ "$ENV_COUNT" -gt 0 ]; then
    echo "✅ Found $ENV_COUNT conda environment files"
else
    echo "⚠️  No conda environment files found in envs/"
fi

# Check if Snakefile is readable
echo ""
echo "Validating Snakefile..."
if python3 -c "
import sys
try:
    with open('Snakefile', 'r') as f:
        content = f.read()
    if 'rule' in content and 'snakemake' in content.lower():
        print('✅ Snakefile appears valid')
        sys.exit(0)
    else:
        print('⚠️  Snakefile may not be a valid Snakemake file')
        sys.exit(1)
except Exception as e:
    print(f'❌ Error reading Snakefile: {e}')
    sys.exit(1)
"; then
    :
else
    echo "❌ Snakefile validation failed"
    exit 1
fi

# Check directory structure
echo ""
echo "Checking directory structure..."
REQUIRED_DIRS=(
    "data"
    "results" 
    "logs"
    "envs"
    "scripts"
    "config"
    "utils"
    "rules"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/"
    else
        echo "⚠️  $dir/ missing (will be created if needed)"
        mkdir -p "$dir"
    fi
done

echo ""
echo "🎉 Docker setup validation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Build the container: make build"
echo "2. Test the setup: make test"
echo "3. Run the pipeline: make run"
echo ""
echo "For Google Cloud deployment:"
echo "1. Set up GCP project: ./scripts/gcp/deploy-cloudrun.sh --project YOUR_PROJECT"
echo "2. Deploy to Cloud Run: automatic with the script"
echo ""
echo "For more information, see: docs/DOCKER_DEPLOYMENT.md"