#!/bin/bash
# Maximum parallelization strategy optimized for 16 vCPU / 64GB RAM
# Efficiently distributes workload across independent stages

set -e

echo "==================================================================="
echo "Maximum parallelization pipeline execution"
echo "Hardware: 16 vCPUs, 64GB RAM"
echo "==================================================================="

# Run all independent stages that don't depend on each other
echo "[Stage 1/7] Metadata & Download (parallel)..."

# Background jobs for independent stages
snakemake --use-conda --cores 16 --jobs 8 metadata_all &
PID1=$!

snakemake --use-conda --cores 16 --jobs 8 download_all &
PID2=$!

# Wait for prerequisites
wait $PID1 $PID2

# QC and Assembly (memory + CPU intensive)
echo "[Stage 2/7] Pre-assembly QC..."
snakemake --use-conda --cores 16 --jobs 8 preassembly_qc_all

echo "[Stage 3/7] Assembly (SPAdes uses 16 threads/job)..."
# SPAdes uses 16 threads per job, so limit to 1 job at a time
snakemake --use-conda --cores 16 --jobs 1 assembly_all

echo "[Stage 4/7] Post-assembly QC (Kraken2 memory-intensive)..."
# Kraken2 uses ~8-16GB RAM per job, limit to 2 parallel jobs
snakemake --use-conda --cores 16 --jobs 2 postassembly_qc_all

# Feature extraction - Run sequentially for max throughput (not parallel)
# Each stage processes ~1186 samples, running sequentially with full resources is faster
echo "[Stage 5/7] Feature extraction..."
echo "  -> AMR analysis (AMRFinder: 4 threads/job, 4 parallel jobs)..."
snakemake --use-conda --cores 16 --jobs 4 amr_analysis_all

echo "  -> SNP analysis (BWA: 4 threads/job, 4 parallel jobs)..."
snakemake --use-conda --cores 16 --jobs 4 snp_analysis_all

# Continue with feature engineering
echo "  -> Feature matrix, selection, batch correction..."
snakemake --use-conda --cores 16 --jobs 8 feature_matrix_all feature_selection_all batch_correction_all

# Dataset preparation - Run sequentially with full resources
# (Previously ran 3 parallel jobs with split resources - inefficient)
echo "[Stage 6/7] Dataset preparation..."
echo "  -> Preparing training data..."
snakemake --use-conda --cores 16 --jobs 8 prepare_training_data_all

echo "  -> Creating k-mer datasets..."
snakemake --use-conda --cores 16 --jobs 1 create_kmer_datasets_all

echo "  -> Creating DNABERT datasets..."
snakemake --use-conda --cores 16 --jobs 1 create_dnabert_datasets_all

# Model training (each model uses 16 threads, run sequentially)
echo "[Stage 7/7] Model training (16 threads each, sequential)..."
echo "  -> XGBoost..."
snakemake --use-conda --cores 16 --jobs 1 train_xgboost_all

echo "  -> LightGBM..."
snakemake --use-conda --cores 16 --jobs 1 train_lightgbm_all

echo "  -> 1D CNN..."
snakemake --use-conda --cores 16 --jobs 1 train_1dcnn_all

echo "  -> Sequence CNN..."
snakemake --use-conda --cores 16 --jobs 1 train_sequence_cnn_all

echo "  -> DNABERT..."
snakemake --use-conda --cores 16 --jobs 1 train_dnabert_all

# Final analysis (lightweight, can run in parallel)
echo "[Stage 8/7] Final analysis (parallel)..."
snakemake --use-conda --cores 8 --jobs 4 interpretability_all &
PID8=$!

snakemake --use-conda --cores 8 --jobs 4 ensemble_analysis_all &
PID9=$!

wait $PID8 $PID9

echo "==================================================================="
echo "Maximum parallelization pipeline completed successfully!"
echo "==================================================================="