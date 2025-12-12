#!/bin/bash
# Setup Google Kubernetes Engine cluster for AMR pipeline

set -e

# Default values
PROJECT_ID=""
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="amr-pipeline-cluster"
MACHINE_TYPE="e2-standard-8"
NUM_NODES=3
MIN_NODES=1
MAX_NODES=10
DISK_SIZE=100
ENABLE_GPU=false
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1

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
        -z|--zone)
            ZONE="$2"
            shift 2
            ;;
        -c|--cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -m|--machine-type)
            MACHINE_TYPE="$2"
            shift 2
            ;;
        -n|--num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --min-nodes)
            MIN_NODES="$2"
            shift 2
            ;;
        --max-nodes)
            MAX_NODES="$2"
            shift 2
            ;;
        --disk-size)
            DISK_SIZE="$2"
            shift 2
            ;;
        --enable-gpu)
            ENABLE_GPU=true
            shift
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -p, --project ID      GCP Project ID (required)"
            echo "  -r, --region REGION   GCP region (default: us-central1)"
            echo "  -z, --zone ZONE       GCP zone (default: us-central1-a)"
            echo "  -c, --cluster NAME    Cluster name (default: amr-pipeline-cluster)"
            echo "  -m, --machine-type    Machine type (default: e2-standard-8)"
            echo "  -n, --num-nodes NUM   Initial node count (default: 3)"
            echo "  --min-nodes NUM       Minimum nodes for autoscaling (default: 1)"
            echo "  --max-nodes NUM       Maximum nodes for autoscaling (default: 10)"
            echo "  --disk-size GB        Boot disk size (default: 100)"
            echo "  --enable-gpu          Enable GPU node pool"
            echo "  --gpu-type TYPE       GPU type (default: nvidia-tesla-t4)"
            echo "  --gpu-count NUM       GPUs per node (default: 1)"
            echo "  -h, --help           Show this help"
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
    echo "Error: Project ID is required"
    exit 1
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    exit 1
fi

echo "Setting up GKE cluster for AMR pipeline"
echo "======================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Zone: $ZONE"
echo "Cluster: $CLUSTER_NAME"
echo "Machine type: $MACHINE_TYPE"
echo "Nodes: $NUM_NODES (min: $MIN_NODES, max: $MAX_NODES)"
echo "GPU enabled: $ENABLE_GPU"
echo ""

# Set the project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    container.googleapis.com \
    compute.googleapis.com \
    artifactregistry.googleapis.com \
    --quiet

# Check if cluster already exists
if gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" &>/dev/null; then
    echo "Cluster $CLUSTER_NAME already exists"
    read -p "Do you want to delete and recreate it? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing cluster..."
        gcloud container clusters delete "$CLUSTER_NAME" --zone="$ZONE" --quiet
    else
        echo "Using existing cluster"
        gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE"
        kubectl cluster-info
        exit 0
    fi
fi

# Create the cluster
echo "Creating GKE cluster..."
CLUSTER_ARGS=(
    "container" "clusters" "create" "$CLUSTER_NAME"
    "--zone" "$ZONE"
    "--machine-type" "$MACHINE_TYPE"
    "--num-nodes" "$NUM_NODES"
    "--enable-autoscaling"
    "--min-nodes" "$MIN_NODES"
    "--max-nodes" "$MAX_NODES"
    "--disk-size" "$DISK_SIZE"
    "--enable-autorepair"
    "--enable-autoupgrade"
    "--enable-ip-alias"
    "--network" "default"
    "--subnetwork" "default"
    "--node-labels" "purpose=amr-pipeline"
    "--node-taints" "amr-pipeline=true:NoSchedule"
    "--scopes" "https://www.googleapis.com/auth/cloud-platform"
    "--quiet"
)

gcloud "${CLUSTER_ARGS[@]}"

# Create GPU node pool if requested
if [ "$ENABLE_GPU" = true ]; then
    echo "Creating GPU node pool..."
    gcloud container node-pools create gpu-pool \
        --cluster="$CLUSTER_NAME" \
        --zone="$ZONE" \
        --machine-type="n1-standard-4" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --num-nodes=1 \
        --min-nodes=0 \
        --max-nodes=3 \
        --enable-autoscaling \
        --enable-autorepair \
        --enable-autoupgrade \
        --node-labels="gpu=true" \
        --node-taints="nvidia.com/gpu=true:NoSchedule" \
        --scopes="https://www.googleapis.com/auth/cloud-platform" \
        --quiet
    
    # Install NVIDIA GPU drivers
    echo "Installing NVIDIA GPU drivers..."
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
fi

# Get cluster credentials
echo "Getting cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE"

# Create namespace
echo "Creating namespace..."
kubectl create namespace amr-pipeline --dry-run=client -o yaml | kubectl apply -f -

# Create storage classes and persistent volumes
echo "Setting up storage..."
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: amr-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  zones: $ZONE
allowVolumeExpansion: true
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: amr-data-pvc
  namespace: amr-pipeline
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: amr-ssd
  resources:
    requests:
      storage: 200Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: amr-results-pvc
  namespace: amr-pipeline
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: amr-ssd
  resources:
    requests:
      storage: 100Gi
EOF

# Create ConfigMap for pipeline configuration
echo "Creating ConfigMap..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: amr-config
  namespace: amr-pipeline
data:
  SNAKEMAKE_CORES: "4"
  SNAKEMAKE_MEMORY: "16G"
  PYTHONUNBUFFERED: "1"
EOF

# Create basic deployment
echo "Creating basic deployment..."
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/amr-pipeline/amr-pipeline:latest"
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: amr-pipeline
  namespace: amr-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: amr-pipeline
  template:
    metadata:
      labels:
        app: amr-pipeline
    spec:
      tolerations:
      - key: "amr-pipeline"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: pipeline
        image: $IMAGE_URL
        imagePullPolicy: Always
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
        envFrom:
        - configMapRef:
            name: amr-config
        volumeMounts:
        - name: data-volume
          mountPath: /pipeline/data
        - name: results-volume
          mountPath: /pipeline/results
        command: ["sleep"]
        args: ["infinity"]
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: amr-data-pvc
      - name: results-volume
        persistentVolumeClaim:
          claimName: amr-results-pvc
EOF

echo ""
echo "✅ GKE cluster setup completed!"
echo ""
echo "Cluster info:"
kubectl cluster-info
echo ""
echo "Nodes:"
kubectl get nodes
echo ""
echo "Deployments:"
kubectl get deployments -n amr-pipeline
echo ""
echo "Next steps:"
echo "1. Build and push your container image:"
echo "   cd /path/to/project && gcloud builds submit --tag $IMAGE_URL"
echo ""
echo "2. Update the deployment:"
echo "   kubectl set image deployment/amr-pipeline pipeline=$IMAGE_URL -n amr-pipeline"
echo ""
echo "3. Run the pipeline:"
echo "   kubectl exec -it deployment/amr-pipeline -n amr-pipeline -- snakemake --help"
echo ""
echo "4. Check logs:"
echo "   kubectl logs -f deployment/amr-pipeline -n amr-pipeline"