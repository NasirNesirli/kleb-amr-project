#!/bin/bash
#SBATCH --job-name=amr_pipeline
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/pipeline_%j.out
#SBATCH --error=logs/slurm/pipeline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

# Set up environment
module purge
module load miniconda3
source activate snakemake

# Set working directory
cd $SLURM_SUBMIT_DIR

# Create log directory
mkdir -p logs/slurm

# Set resource limits based on Slurm allocation
export SNAKEMAKE_CORES=$SLURM_CPUS_PER_TASK
export SNAKEMAKE_MEM=${SLURM_MEM_PER_NODE}

echo "Starting AMR K. pneumoniae prediction pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Cores allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"

# Run the pipeline with cluster configuration
snakemake \
    --profile slurm \
    --cores $SLURM_CPUS_PER_TASK \
    --resources mem_gb=$((SLURM_MEM_PER_NODE/1024)) \
    --use-conda \
    --conda-frontend conda \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    2>&1 | tee logs/slurm/snakemake_${SLURM_JOB_ID}.log

echo "End time: $(date)"
echo "Pipeline completed with exit code: $?"