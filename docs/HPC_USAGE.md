# HPC/Slurm Deployment Guide

## 🖥️ Running the AMR Pipeline on HPC Clusters

This guide explains how to run the K. pneumoniae AMR prediction pipeline on High Performance Computing (HPC) clusters using Slurm.

## 📋 Prerequisites

### System Requirements
- **Slurm Workload Manager** installed on the cluster
- **Miniconda/Anaconda** available as a module or installed
- **8+ CPU cores** per node (16+ recommended)
- **64+ GB RAM** per node (128+ GB recommended for deep learning)
- **1+ TB storage** for full dataset
- **GPU support** (optional but recommended for deep learning models)

### Account Setup
```bash
# Check your Slurm account and partitions
sinfo                    # View available partitions
squeue -u $USER         # View your job queue
sacct -u $USER          # View job history
```

## ⚙️ Configuration

### 1. Update Cluster Settings
Edit the Slurm configuration files to match your cluster:

```bash
# Edit slurm/cluster.yaml
vim slurm/cluster.yaml

# Update these fields:
__default__:
  account: "YOUR_ACCOUNT_NAME"      # Your Slurm account
  partition: "YOUR_PARTITION"       # Default partition (e.g., "compute", "standard")
  
# For GPU jobs, specify GPU partition:
train_dnabert:
  partition: "gpu"                  # GPU partition name
  gres: "gpu:1"                     # GPU resource request
```

### 2. Update Job Scripts
Edit the main submission script:

```bash
vim scripts/run_pipeline_slurm.sh

# Update these SBATCH directives:
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --mail-user=YOUR_EMAIL
```

## 🚀 Usage Examples

### Option 1: Full Pipeline (Single Job)
```bash
# Submit entire pipeline as one job
sbatch scripts/run_pipeline_slurm.sh

# Monitor job progress
squeue -u $USER
tail -f logs/slurm/pipeline_JOBID.out
```

### Option 2: Stage-by-Stage Execution
```bash
# Create stage-specific scripts
./scripts/run_stages_slurm.sh create_scripts

# Run stages sequentially with dependency management
./scripts/run_stages_slurm.sh run_sequential

# Or submit individual stages manually:
sbatch scripts/slurm_stages/01_preprocessing.sh
# Wait for completion, then:
sbatch scripts/slurm_stages/02_features.sh
# etc...
```

### Option 3: Interactive Development
```bash
# Request interactive session
srun --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
     --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash

# Load conda and activate environment
module load miniconda3
conda activate snakemake

# Run specific rules interactively
snakemake --profile slurm --cores 8 process_metadata
snakemake --profile slurm --cores 8 train_xgboost
```

## 📊 Resource Requirements by Stage

| Stage | Cores | Memory | Time | Storage | GPU |
|-------|-------|--------|------|---------|-----|
| **Metadata Processing** | 1 | 4GB | 30min | 1GB | No |
| **Data Download** | 2 | 8GB | 4hrs | 100GB | No |
| **QC & Assembly** | 16 | 64GB | 8hrs | 200GB | No |
| **Feature Extraction** | 8 | 32GB | 4hrs | 50GB | No |
| **Tree Models** | 16 | 64GB | 6hrs | 10GB | No |
| **Deep Learning** | 8 | 64GB | 24hrs | 20GB | Yes |
| **Interpretability** | 8 | 32GB | 4hrs | 5GB | No |

## 🎯 Cluster-Specific Examples

### SLAC/Stanford
```bash
# SLAC Computing cluster
#SBATCH --partition=shared
#SBATCH --account=your_group
#SBATCH --qos=normal

module load conda
conda activate snakemake-7.32.4
```

### NERSC (Perlmutter)
```bash
# NERSC Perlmutter
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=your_project

module load conda
conda activate snakemake
```

### Generic Academic HPC
```bash
# Most academic clusters
#SBATCH --partition=compute
#SBATCH --account=pi_username

module load Miniconda3
source activate snakemake
```

## 💡 Optimization Tips

### 1. **Parallel Job Submission**
```bash
# Snakemake automatically parallelizes independent jobs
snakemake --profile slurm --jobs 50 --cores 200
```

### 2. **Resource Monitoring**
```bash
# Monitor resource usage
seff JOBID                # Job efficiency report
sstat JOBID              # Real-time job stats
sacct -j JOBID --format=JobID,JobName,MaxRSS,Elapsed,CPUTime
```

### 3. **Storage Optimization**
```bash
# Use fast local storage for temporary files
export TMPDIR=/local/scratch/$SLURM_JOB_ID
mkdir -p $TMPDIR

# Clean up temporary files
trap "rm -rf $TMPDIR" EXIT
```

### 4. **Checkpoint and Resume**
```bash
# Snakemake automatically handles checkpointing
# Resume failed runs:
sbatch scripts/run_pipeline_slurm.sh  # Will continue from last checkpoint
```

## 🔧 Troubleshooting

### Common Issues

#### 1. **Job Timeout**
```bash
# Increase time limits in slurm/cluster.yaml
train_dnabert:
  time: "48:00:00"  # Increase for large datasets
```

#### 2. **Memory Exceeded**
```bash
# Check memory usage and increase allocation
train_xgboost:
  mem: "128G"  # Increase memory for large feature sets
```

#### 3. **GPU Not Available**
```bash
# Check GPU availability
sinfo --format="%.15N %.10c %.10m %.25f %.10G"

# Request specific GPU type
#SBATCH --gres=gpu:v100:1
```

#### 4. **Conda Environment Issues**
```bash
# Create environments on compute nodes
srun --pty bash
conda env create -f envs/snakemake.yaml
```

### Debugging Commands
```bash
# Check job details
scontrol show job JOBID

# View job logs
less logs/slurm/pipeline_JOBID.out
less logs/slurm/rule_JOBID.err

# Test Slurm profile
snakemake --profile slurm --dry-run
```

## 📈 Performance Expectations

### Small Dataset (120 samples - current)
- **Total time**: 8-12 hours
- **Total cost**: $50-100 (on cloud HPC)
- **Peak memory**: 64GB
- **Storage**: 200GB

### Large Dataset (1000+ samples)
- **Total time**: 24-48 hours
- **Peak memory**: 128GB
- **Storage**: 1-2TB
- **Recommended**: Multi-node with GPU support

## 🎯 Best Practices

1. **Start Small**: Test with a subset of samples first
2. **Monitor Resources**: Use `seff` to check job efficiency
3. **Use Checkpoints**: Snakemake handles automatic resumption
4. **Profile Code**: Identify bottlenecks before scaling up
5. **Use GPUs**: Essential for deep learning models (3-5x speedup)
6. **Clean Temp Files**: Set up automatic cleanup in job scripts

## 📞 Support

For cluster-specific questions:
- Check your HPC center's documentation
- Contact your system administrator
- Use cluster-specific Slack/forums
- Monitor cluster status pages

For pipeline questions:
- Check `results/` directory for outputs
- Review Snakemake logs in `logs/`
- Use `--dry-run` to test configurations