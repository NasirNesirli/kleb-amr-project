#!/bin/bash
# Maximum parallelization strategy - runs independent stages simultaneously

set -e

echo "Maximum parallelization pipeline execution..."

# Run all independent stages that don't depend on each other
echo "Running all possible parallel stages..."

# Background jobs for independent stages
snakemake --use-conda --cores 4 --jobs 2 metadata_all &
PID1=$!

snakemake --use-conda --cores 12 --jobs 6 download_all &
PID2=$!

# Wait for prerequisites
wait $PID1 $PID2

# QC and Assembly (high resource usage)
snakemake --use-conda --cores 16 --jobs 4 preassembly_qc_all assembly_all postassembly_qc_all

# Feature extraction (can run in parallel)
snakemake --use-conda --cores 8 --jobs 4 amr_analysis_all &
PID3=$!

snakemake --use-conda --cores 8 --jobs 4 snp_analysis_all &
PID4=$!

wait $PID3 $PID4

# Continue with feature engineering
snakemake --use-conda --cores 16 --jobs 6 feature_matrix_all feature_selection_all batch_correction_all

# Dataset preparation (parallel)
snakemake --use-conda --cores 5 --jobs 3 prepare_training_data_all &
PID5=$!

snakemake --use-conda --cores 5 --jobs 3 create_kmer_datasets_all &
PID6=$!

snakemake --use-conda --cores 6 --jobs 3 create_dnabert_datasets_all &
PID7=$!

wait $PID5 $PID6 $PID7

# Model training (sequential for resource management)
echo "Training models sequentially to manage resources..."
snakemake --use-conda --cores 16 train_xgboost_all
snakemake --use-conda --cores 16 train_lightgbm_all
snakemake --use-conda --cores 16 train_1dcnn_all
snakemake --use-conda --cores 16 train_sequence_cnn_all
snakemake --use-conda --cores 16 train_dnabert_all

# Final analysis (can run in parallel)
snakemake --use-conda --cores 8 --jobs 2 interpretability_all &
PID8=$!

snakemake --use-conda --cores 8 --jobs 2 ensemble_analysis_all &
PID9=$!

wait $PID8 $PID9

echo "Maximum parallelization pipeline completed!"