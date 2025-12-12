# Google Cloud Resource Estimation & Recommendations

## 📊 Current Dataset Analysis

Based on the existing project structure:

| Component | Current Size | Sample Count | Avg per Sample |
|-----------|--------------|--------------|----------------|
| **Raw FASTQ (processed)** | 86GB | 120 samples | ~717MB/sample |
| **Assemblies** | 652MB | 120 samples | ~5.4MB/sample |
| **Results** | 2.6GB | 120 samples | ~22MB/sample |
| **Total Project** | 109GB | 120 samples | ~908MB/sample |

## 🎯 Full Dataset Projections (1000+ samples)

### Proposed Full Dataset: 1,100 samples
- **Training**: 1,000 samples (pre-2024)
- **Test**: 100 samples (2024-25)

| Component | Projected Size | Calculation |
|-----------|----------------|-------------|
| **Raw FASTQ** | **750-800GB** | 1,100 × 717MB ≈ 788GB |
| **Assemblies** | **6GB** | 1,100 × 5.4MB ≈ 6GB |
| **Intermediate files** | **100-150GB** | SNP calling, feature extraction |
| **Model results** | **25GB** | 1,100 × 22MB ≈ 24GB |
| **Working space** | **200GB** | Temporary files, conda envs |
| **Total Storage** | **🎯 1.1-1.3TB** | Conservative estimate |

## ☁️ Google Cloud Recommendations

### 💰 **Cost-Optimized Configuration** (Recommended)

#### Cloud Run (Serverless - Best for small-medium datasets)
```bash
# Perfect for current 120-sample dataset
Memory: 16GB
CPU: 4 vCPUs
Timeout: 3600s
Max instances: 3
Estimated cost: $0.50-2.00 per run
```

#### Compute Engine (Best for full dataset)
```bash
# Optimal balance of cost and performance
Machine: e2-standard-8
vCPUs: 8
Memory: 32GB
Boot disk: 100GB SSD
Data disk: 1.5TB SSD (for safety margin)
Network: Standard
Preemptible: Yes (60-90% cost savings)

Estimated cost: $4-6/day (preemptible) vs $15-20/day (regular)
```

#### Cloud Storage Strategy
```bash
# Multi-tier storage for cost optimization
Raw FASTQ: Nearline Storage (30-day access) - $0.010/GB/month
Working data: SSD Persistent Disk - $0.17/GB/month  
Results/Models: Standard Storage - $0.020/GB/month
Archives: Archive Storage (1-year) - $0.0012/GB/month

Estimated storage cost: $180-250/month for full dataset
```

### ⚡ **Performance-Optimized Configuration**

#### For time-sensitive analysis:
```bash
Machine: c2-standard-16
vCPUs: 16
Memory: 64GB
Boot disk: 200GB SSD
Data disk: 2TB Local SSD (fastest I/O)
Network: Premium

Estimated cost: $25-35/day
Performance gain: 3-4x faster than cost-optimized
```

### 🧠 **Deep Learning Optimization**

#### For DNABERT and CNN training:
```bash
Machine: n1-standard-8 + 1x NVIDIA T4
vCPUs: 8
Memory: 30GB
GPU: 1x T4 (16GB VRAM)
Storage: 1.5TB SSD

Estimated cost: $8-12/day
Training speedup: 5-10x for deep learning models
```

## 📋 **Stage-Specific Resource Requirements**

| Pipeline Stage | CPU | Memory | Time | Optimal Machine |
|----------------|-----|--------|------|-----------------|
| **Download (1-2)** | 2-4 cores | 8GB | 2-6 hours | e2-standard-4 |
| **QC/Assembly (3-4)** | 8-16 cores | 32GB | 6-12 hours | c2-standard-16 |
| **Feature extraction (6-8)** | 4-8 cores | 16GB | 2-4 hours | e2-standard-8 |
| **Tree models (14-15)** | 8 cores | 32GB | 1-2 hours | e2-standard-8 |
| **Deep learning (16-18)** | 8 cores + GPU | 32GB | 4-8 hours | n1-standard-8 + T4 |
| **Interpretability (19)** | 4 cores | 16GB | 1 hour | e2-standard-4 |

## 🎯 **Recommended Deployment Strategy**

### Option 1: Single Large Instance (Simplest)
```bash
# Run entire pipeline on one machine
Machine: e2-standard-8 (preemptible)
Storage: 1.5TB SSD
Total time: 12-18 hours
Cost: ~$6-8 per full run
```

### Option 2: Stage-Optimized (Most efficient)
```bash
# Different machines for different stages
Preprocessing: c2-standard-16 (4-6 hours) → $15
ML Training: e2-standard-8 (3-4 hours) → $4  
Deep Learning: n1-standard-8 + T4 (2-3 hours) → $8
Total: ~$27, but 2x faster execution
```

### Option 3: Kubernetes Auto-scaling (Production)
```bash
# Auto-scaling based on workload
Node pool 1: e2-standard-4 (min: 1, max: 10)
Node pool 2: n1-standard-8 + T4 (min: 0, max: 3)
Storage: GCS + persistent volumes
Cost: $50-200/month depending on usage
```

## 💡 **Cost Optimization Tips**

### 1. **Use Preemptible Instances** (60-90% savings)
```bash
./scripts/gcp/deploy-cloudrun.sh --preemptible
# Savings: $15/day → $4/day
```

### 2. **Tiered Storage Strategy**
```bash
# Raw data → Nearline (accessed monthly)
gsutil cp -m data/processed/* gs://bucket-nearline/

# Working data → SSD (active processing)  
# Results → Standard (frequent access)
# Old results → Archive (long-term storage)
```

### 3. **Scheduled Execution**
```bash
# Run during off-peak hours (cheaper)
# Use Cloud Scheduler for automated runs
# Scale down during idle periods
```

### 4. **Regional Optimization**
```bash
# Choose cheaper regions
us-central1: Standard pricing
us-west2: 10-20% higher
europe-west4: 15-25% higher

# Recommended: us-central1 (Iowa)
```

## 🚀 **Quick Start Commands**

### Cost-Optimized Setup
```bash
# Deploy cost-optimized Cloud Run
./scripts/gcp/deploy-cloudrun.sh \
  --project YOUR_PROJECT \
  --memory 16Gi \
  --cpu 4 \
  --region us-central1

# Create GKE with preemptible nodes  
./scripts/gcp/setup-gke.sh \
  --project YOUR_PROJECT \
  --machine-type e2-standard-8 \
  --preemptible
```

### Performance Setup
```bash
# High-performance Compute Engine
gcloud compute instances create amr-pipeline-fast \
  --zone us-central1-a \
  --machine-type c2-standard-16 \
  --boot-disk-size 200GB \
  --create-disk size=2TB,type=pd-ssd \
  --preemptible

# GPU-enabled for deep learning
gcloud compute instances create amr-pipeline-gpu \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --maintenance-policy TERMINATE \
  --preemptible
```

## 📊 **Cost Summary Table**

| Configuration | Storage | Compute | Total/Month | Use Case |
|---------------|---------|---------|-------------|----------|
| **Development** | 100GB | Cloud Run | $20-40 | Small datasets, testing |
| **Research** | 1.5TB | e2-standard-8 (preemptible) | $180-250 | Full 1000-sample dataset |
| **Production** | 2TB | Auto-scaling GKE | $300-500 | Clinical deployment |
| **High-Performance** | 2TB | c2-standard-16 + GPU | $600-800 | Time-critical analysis |

## 🎯 **Final Recommendation**

### **For your 1000+ sample project:**

```bash
Configuration: e2-standard-8 (preemptible) + 1.5TB SSD
Estimated monthly cost: $200-300
Execution time: 12-18 hours per full run
Storage strategy: Nearline for raw data, SSD for active work

# Deploy command:
./scripts/gcp/deploy-cloudrun.sh \
  --project YOUR_PROJECT \
  --memory 32Gi \
  --cpu 8 \
  --timeout 7200 \
  --region us-central1
```

This provides the **optimal balance of cost and performance** for your AMR prediction pipeline while maintaining the flexibility to scale up for time-sensitive analyses.