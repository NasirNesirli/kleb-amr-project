#!/bin/bash
# Resource-aware pipeline execution based on stage requirements

# System resources
TOTAL_CORES=16
TOTAL_RAM=64

run_stage() {
    local stage_name=$1
    local cores=$2
    local jobs=$3
    local targets=$4
    
    echo "Running $stage_name with $cores cores, $jobs parallel jobs..."
    snakemake --use-conda --cores $cores --jobs $jobs $targets
}

# Light I/O stages (high parallelization)
run_stage "Metadata Processing" 4 8 "metadata_all"
run_stage "Data Download" 12 6 "download_all"

# CPU-intensive stages (moderate parallelization)  
run_stage "Pre-assembly QC" 16 4 "preassembly_qc_all"
run_stage "Assembly" 16 2 "assembly_all"  # Reduced jobs due to high memory per job
run_stage "Post-assembly QC" 16 4 "postassembly_qc_all"

# Analysis stages (balanced)
run_stage "AMR Analysis" 8 4 "amr_analysis_all"
run_stage "SNP Analysis" 8 4 "snp_analysis_all"
run_stage "Feature Matrix" 16 6 "feature_matrix_all"
run_stage "Feature Selection" 16 4 "feature_selection_all"
run_stage "Batch Correction" 16 4 "batch_correction_all"

# Dataset preparation (memory-intensive)
run_stage "Training Data Prep" 8 4 "prepare_training_data_all"
run_stage "K-mer Datasets" 8 2 "create_kmer_datasets_all"
run_stage "DNABERT Datasets" 8 2 "create_dnabert_datasets_all"

# Model training (resource-intensive, run sequentially)
echo "Training models with full resources..."
run_stage "XGBoost Training" 16 1 "train_xgboost_all"
run_stage "LightGBM Training" 16 1 "train_lightgbm_all" 
run_stage "1D CNN Training" 16 1 "train_1dcnn_all"
run_stage "Sequence CNN Training" 16 1 "train_sequence_cnn_all"
run_stage "DNABERT Training" 16 1 "train_dnabert_all"

# Final analysis
run_stage "Interpretability Analysis" 8 4 "interpretability_all"
run_stage "Ensemble Analysis" 8 4 "ensemble_analysis_all"

echo "Resource-aware pipeline completed!"