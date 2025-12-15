#!/bin/bash
# AMR Pipeline Stage-specific Slurm Execution Scripts

# Function to submit a job and wait for completion
submit_and_wait() {
    local script_path=$1
    local job_name=$2
    
    echo "Submitting $job_name..."
    job_id=$(sbatch --parsable "$script_path")
    echo "Job ID: $job_id"
    
    # Wait for job completion
    while true; do
        status=$(sacct -j "$job_id" --format=State --noheader --parsable2 | head -1)
        case $status in
            "COMPLETED")
                echo "$job_name completed successfully"
                break
                ;;
            "FAILED"|"CANCELLED"|"TIMEOUT"|"OUT_OF_MEMORY")
                echo "$job_name failed with status: $status"
                exit 1
                ;;
            *)
                echo "$job_name status: $status. Waiting..."
                sleep 30
                ;;
        esac
    done
}

# Create stage-specific scripts
create_stage_scripts() {
    mkdir -p scripts/slurm_stages
    
    # Preprocessing stages (1-5)
    cat > scripts/slurm_stages/01_preprocessing.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=amr_preprocess
#SBATCH --partition=YOUR_PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/preprocessing_%j.out
#SBATCH --error=logs/slurm/preprocessing_%j.err

cd $SLURM_SUBMIT_DIR
snakemake --profile slurm preprocess --cores $SLURM_CPUS_PER_TASK
EOF

    # Feature extraction (6-10)
    cat > scripts/slurm_stages/02_features.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=amr_features
#SBATCH --partition=YOUR_PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/features_%j.out
#SBATCH --error=logs/slurm/features_%j.err

cd $SLURM_SUBMIT_DIR
snakemake --profile slurm feature_extraction --cores $SLURM_CPUS_PER_TASK
EOF

    # Tree models (14-15)
    cat > scripts/slurm_stages/03_tree_models.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=amr_tree_models
#SBATCH --partition=YOUR_PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/tree_models_%j.out
#SBATCH --error=logs/slurm/tree_models_%j.err

cd $SLURM_SUBMIT_DIR
snakemake --profile slurm tree_models --cores $SLURM_CPUS_PER_TASK
EOF

    # Deep learning models (16-18)
    cat > scripts/slurm_stages/04_dl_models.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=amr_dl_models
#SBATCH --partition=gpu  # Use GPU partition if available
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/dl_models_%j.out
#SBATCH --error=logs/slurm/dl_models_%j.err

cd $SLURM_SUBMIT_DIR
snakemake --profile slurm dl_models --cores $SLURM_CPUS_PER_TASK
EOF

    # Interpretability analysis (19)
    cat > scripts/slurm_stages/05_interpretability.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=amr_interpret
#SBATCH --partition=YOUR_PARTITION
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/interpretability_%j.out
#SBATCH --error=logs/slurm/interpretability_%j.err

cd $SLURM_SUBMIT_DIR
snakemake --profile slurm interpretability --cores $SLURM_CPUS_PER_TASK
EOF

    chmod +x scripts/slurm_stages/*.sh
    echo "Created stage-specific Slurm scripts in scripts/slurm_stages/"
}

# Main execution
case $1 in
    "create_scripts")
        create_stage_scripts
        ;;
    "run_sequential")
        create_stage_scripts
        submit_and_wait "scripts/slurm_stages/01_preprocessing.sh" "preprocessing"
        submit_and_wait "scripts/slurm_stages/02_features.sh" "features"
        submit_and_wait "scripts/slurm_stages/03_tree_models.sh" "tree_models"
        submit_and_wait "scripts/slurm_stages/04_dl_models.sh" "dl_models"
        submit_and_wait "scripts/slurm_stages/05_interpretability.sh" "interpretability"
        echo "All stages completed successfully!"
        ;;
    *)
        echo "Usage: $0 {create_scripts|run_sequential}"
        echo "  create_scripts  : Create stage-specific Slurm submission scripts"
        echo "  run_sequential  : Run all stages sequentially with dependency management"
        ;;
esac