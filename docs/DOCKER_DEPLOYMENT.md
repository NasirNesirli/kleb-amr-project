# Docker Deployment Guide

This guide covers containerized deployment options for the AMR K. pneumoniae prediction pipeline, including local Docker, Google Cloud, AWS, and HPC environments.

## 🐳 Local Docker Deployment

### Prerequisites
- Docker 20.10+ and Docker Compose v2.0+
- 8+ GB RAM (16+ GB recommended)
- 4+ CPU cores (8+ recommended)

### Quick Start

```bash
# Build the container
make build

# Test the setup
make test

# Run the pipeline
make run

# Interactive shell
make shell
```

### Available Commands

| Command | Description |
|---------|-------------|
| `make build` | Build production container |
| `make build-dev` | Build development container |
| `make run` | Run full pipeline |
| `make dry-run` | Check pipeline without execution |
| `make shell` | Interactive shell |
| `make clean` | Clean up containers/images |

### Custom Data Directories

```bash
# Run with custom data paths
./scripts/docker/run.sh \
  --data-dir /path/to/your/data \
  --results-dir /path/to/results \
  --cores 8 \
  --memory 16G
```

### Development with Docker Compose

```bash
# Start development environment
docker-compose up amr-pipeline-dev

# Run specific stages
docker-compose run amr-pipeline --target tree_models
```

## ☁️ Google Cloud Platform Deployment

### Option 1: Cloud Run (Serverless)

#### Setup
```bash
# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create amr-pipeline \
  --repository-format=docker \
  --location=us-central1
```

#### Build and Deploy
```bash
# Configure Docker for GCP
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and push to Artifact Registry
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/amr-pipeline/amr-pipeline:latest

# Deploy to Cloud Run
gcloud run deploy amr-pipeline \
  --image us-central1-docker.pkg.dev/PROJECT_ID/amr-pipeline/amr-pipeline:latest \
  --platform managed \
  --region us-central1 \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600 \
  --max-instances 10 \
  --allow-unauthenticated
```

#### Cloud Run YAML Configuration
```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: amr-pipeline
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/memory: "16Gi"
        run.googleapis.com/cpu: "4"
        run.googleapis.com/timeout: "3600s"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 1
      containers:
      - image: us-central1-docker.pkg.dev/PROJECT_ID/amr-pipeline/amr-pipeline:latest
        resources:
          limits:
            memory: "16Gi"
            cpu: "4"
        env:
        - name: SNAKEMAKE_CORES
          value: "4"
        - name: SNAKEMAKE_MEMORY
          value: "16G"
        volumeMounts:
        - name: data-volume
          mountPath: /pipeline/data
        - name: results-volume
          mountPath: /pipeline/results
      volumes:
      - name: data-volume
        csi:
          driver: gcsfuse.csi.storage.gke.io
          volumeAttributes:
            bucketName: your-data-bucket
      - name: results-volume
        csi:
          driver: gcsfuse.csi.storage.gke.io
          volumeAttributes:
            bucketName: your-results-bucket
```

### Option 2: Google Kubernetes Engine (GKE)

#### Setup GKE Cluster
```bash
# Create cluster
gcloud container clusters create amr-cluster \
  --zone us-central1-a \
  --machine-type e2-standard-8 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials amr-cluster --zone us-central1-a
```

#### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: amr-pipeline
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
      containers:
      - name: pipeline
        image: us-central1-docker.pkg.dev/PROJECT_ID/amr-pipeline/amr-pipeline:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
        env:
        - name: SNAKEMAKE_CORES
          value: "4"
        volumeMounts:
        - name: data-pvc
          mountPath: /pipeline/data
        - name: results-pvc
          mountPath: /pipeline/results
      volumes:
      - name: data-pvc
        persistentVolumeClaim:
          claimName: data-pvc
      - name: results-pvc
        persistentVolumeClaim:
          claimName: results-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: results-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
```

### Option 3: Compute Engine VM

#### Create VM with Container
```bash
# Create VM with container
gcloud compute instances create-with-container amr-pipeline-vm \
  --zone us-central1-a \
  --machine-type e2-standard-8 \
  --boot-disk-size 100GB \
  --container-image us-central1-docker.pkg.dev/PROJECT_ID/amr-pipeline/amr-pipeline:latest \
  --container-restart-policy always \
  --container-env SNAKEMAKE_CORES=4 \
  --metadata startup-script='#!/bin/bash
    # Mount additional disk for data
    mkfs.ext4 /dev/sdb
    mkdir -p /mnt/data
    mount /dev/sdb /mnt/data
    '

# SSH into VM
gcloud compute ssh amr-pipeline-vm --zone us-central1-a
```

### Option 4: Cloud Build for CI/CD

#### Cloud Build Configuration
```yaml
# cloudbuild.yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'us-central1-docker.pkg.dev/$PROJECT_ID/amr-pipeline/amr-pipeline:$BUILD_ID', '.']

- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'us-central1-docker.pkg.dev/$PROJECT_ID/amr-pipeline/amr-pipeline:$BUILD_ID']

- name: 'gcr.io/cloud-builders/gcloud'
  args: 
  - 'run'
  - 'deploy'
  - 'amr-pipeline'
  - '--image=us-central1-docker.pkg.dev/$PROJECT_ID/amr-pipeline/amr-pipeline:$BUILD_ID'
  - '--region=us-central1'
  - '--platform=managed'
  - '--memory=16Gi'
  - '--cpu=4'

images:
- 'us-central1-docker.pkg.dev/$PROJECT_ID/amr-pipeline/amr-pipeline:$BUILD_ID'

options:
  machineType: 'E2_HIGHCPU_8'
  diskSizeGb: '100'
```

#### Trigger Build
```bash
# Manual build
gcloud builds submit --config cloudbuild.yaml

# Set up GitHub trigger
gcloud builds triggers create github \
  --repo-name amr-kpneumoniae-prediction \
  --repo-owner your-username \
  --branch-pattern "^main$" \
  --build-config cloudbuild.yaml
```

## 🗂️ Data Management

### Google Cloud Storage Integration

```bash
# Create buckets
gsutil mb gs://your-amr-data-bucket
gsutil mb gs://your-amr-results-bucket

# Upload data
gsutil -m cp -r data/* gs://your-amr-data-bucket/

# Mount in container
docker run \
  -v /path/to/gcs-fuse:/pipeline/data:ro \
  -v /path/to/gcs-fuse:/pipeline/results:rw \
  amr-pipeline:latest
```

### Cloud Storage FUSE Setup
```bash
# Install gcsfuse
curl -L -O https://github.com/GoogleCloudPlatform/gcsfuse/releases/latest/download/gcsfuse_amd64.deb
sudo dpkg -i gcsfuse_amd64.deb

# Mount buckets
mkdir -p /mnt/data /mnt/results
gcsfuse your-amr-data-bucket /mnt/data
gcsfuse your-amr-results-bucket /mnt/results
```

## 📊 Monitoring and Logging

### Cloud Logging
```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=amr-pipeline"

# Create log-based metrics
gcloud logging metrics create amr_pipeline_errors \
  --description="AMR pipeline error count" \
  --log-filter='resource.type=cloud_run_revision AND resource.labels.service_name=amr-pipeline AND severity>=ERROR'
```

### Cloud Monitoring Dashboard
```json
{
  "displayName": "AMR Pipeline Monitoring",
  "dashboardFilters": [],
  "widgets": [
    {
      "title": "Pipeline Execution Time",
      "xyChart": {
        "dataSets": [{
          "timeSeriesQuery": {
            "timeSeriesFilter": {
              "filter": "resource.type=cloud_run_revision",
              "aggregation": {
                "alignmentPeriod": "60s",
                "perSeriesAligner": "ALIGN_RATE"
              }
            }
          }
        }]
      }
    }
  ]
}
```

## 💰 Cost Optimization

### Resource Sizing Guidelines

| Pipeline Stage | CPU | Memory | Duration | GCP Machine Type |
|----------------|-----|---------|----------|------------------|
| Preprocessing | 2-4 cores | 8GB | 30-60 min | e2-standard-4 |
| Tree Models | 4-8 cores | 16GB | 15-30 min | e2-standard-8 |
| Deep Learning | 4-8 cores + GPU | 32GB | 60-120 min | n1-standard-8 + T4 |
| Interpretability | 2-4 cores | 8GB | 10-20 min | e2-standard-4 |

### Preemptible Instances
```bash
# Use preemptible VMs for cost savings
gcloud compute instances create amr-pipeline-preemptible \
  --zone us-central1-a \
  --machine-type e2-standard-8 \
  --preemptible \
  --boot-disk-size 50GB \
  --container-image us-central1-docker.pkg.dev/PROJECT_ID/amr-pipeline/amr-pipeline:latest
```

### Scheduled Execution
```yaml
# Cloud Scheduler job
name: projects/PROJECT_ID/locations/us-central1/jobs/amr-pipeline-daily
schedule: "0 2 * * *"  # Daily at 2 AM
timeZone: "America/New_York"
httpTarget:
  uri: https://amr-pipeline-SERVICE-us-central1.a.run.app
  httpMethod: POST
  headers:
    Content-Type: application/json
  body: eyJjb3JlcyI6IDQsICJ0YXJnZXQiOiAiZnVsbF9waXBlbGluZSJ9  # base64 encoded JSON
```

## 🔒 Security Best Practices

### Service Account Setup
```bash
# Create service account
gcloud iam service-accounts create amr-pipeline-sa \
  --description="AMR pipeline service account"

# Grant minimal permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:amr-pipeline-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Use in Cloud Run
gcloud run deploy amr-pipeline \
  --service-account amr-pipeline-sa@PROJECT_ID.iam.gserviceaccount.com
```

### Secret Management
```bash
# Store sensitive data in Secret Manager
echo -n "your-api-key" | gcloud secrets create amr-api-key --data-file=-

# Access in container
gcloud run deploy amr-pipeline \
  --set-secrets="API_KEY=amr-api-key:latest"
```

## 🚀 Example Deployment Scripts

### Complete GCP Deployment
```bash
#!/bin/bash
# deploy-gcp.sh

set -e

PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="amr-pipeline"

# Build and push
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600 \
  --max-instances 5 \
  --allow-unauthenticated

echo "Deployment completed!"
echo "Service URL: $(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')"
```

This comprehensive deployment guide provides multiple options for running your AMR pipeline in Google Cloud Platform, from serverless Cloud Run to full Kubernetes deployments.