#!/bin/bash
# Optimized pipeline execution with full resource utilization

set -e

echo "Starting optimized AMR pipeline execution..."

# Stage 1: Data Preparation (can run in parallel)
echo "Stage 1: Data preparation..."
snakemake --use-conda --cores 16 --jobs 4 \
    metadata_all download_all &

wait  # Wait for data prep to complete

# Stage 2: QC and Assembly (parallel processing)
echo "Stage 2: QC and Assembly..."
snakemake --use-conda --cores 16 --jobs 6 \
    preassembly_qc_all assembly_all postassembly_qc_all

# Stage 3: Feature Engineering (parallel where possible)
echo "Stage 3: Feature extraction and engineering..."
snakemake --use-conda --cores 16 --jobs 8 \
    amr_analysis_all snp_analysis_all feature_matrix_all \
    feature_selection_all batch_correction_all

# Stage 4: Dataset Preparation (parallel)
echo "Stage 4: Dataset preparation..."
snakemake --use-conda --cores 16 --jobs 6 \
    prepare_training_data_all create_kmer_datasets_all create_dnabert_datasets_all

# Stage 5: Model Training (resource-intensive, limit parallelization)
echo "Stage 5: Tree model training..."
snakemake --use-conda --cores 16 --jobs 2 \
    train_xgboost_all train_lightgbm_all

echo "Stage 6: Deep learning training (GPU-intensive)..."
snakemake --use-conda --cores 16 --jobs 1 \
    train_1dcnn_all train_sequence_cnn_all train_dnabert_all

# Stage 6: Analysis (can run in parallel)
echo "Stage 7: Final analysis..."
snakemake --use-conda --cores 16 --jobs 4 \
    interpretability_all ensemble_analysis_all

echo "Pipeline completed successfully!"