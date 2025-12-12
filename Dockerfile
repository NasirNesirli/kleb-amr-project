# AMR K. pneumoniae Prediction Pipeline - Docker Container
# Multi-stage build for optimized container size and caching

# Stage 1: Base environment with system dependencies
FROM mambaforge/mambaforge:latest AS base

# Set working directory
WORKDIR /pipeline

# Install system dependencies and tools
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Update conda and install mamba for faster dependency resolution
RUN conda update -n base -c defaults conda && \
    conda install -n base -c conda-forge mamba

# Stage 2: Pipeline setup
FROM base AS pipeline-setup

# Install Snakemake and core bioinformatics tools
RUN mamba install -c conda-forge -c bioconda \
    snakemake>=7.0.0 \
    conda-build \
    && conda clean -afy

# Copy environment files first (for better Docker layer caching)
COPY envs/ ./envs/
COPY config/ ./config/

# Pre-create conda environments to speed up pipeline execution
# This step caches the environment creation for faster container startup
RUN snakemake --use-conda --conda-create-envs-only --cores 1 || true

# Stage 3: Production container
FROM pipeline-setup AS production

# Copy the entire pipeline
COPY . .

# Set up proper permissions
RUN chmod +x scripts/*.py && \
    chmod 755 /pipeline

# Create directories for mounted volumes
RUN mkdir -p /pipeline/data /pipeline/results /pipeline/logs

# Set environment variables
ENV SNAKEMAKE_CONDA_PREFIX=/opt/conda/envs
ENV PYTHONPATH=/pipeline:$PYTHONPATH
ENV PATH=/pipeline/scripts:$PATH

# Create non-root user for security
RUN groupadd -r amruser && useradd -r -g amruser -d /pipeline amruser
RUN chown -R amruser:amruser /pipeline
USER amruser

# Set default command to show help
ENTRYPOINT ["snakemake", "--use-conda", "--conda-frontend", "mamba"]
CMD ["--help"]

# Health check to verify container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD snakemake --version || exit 1

# Labels for metadata
LABEL maintainer="MSc Project - AMR Prediction"
LABEL version="1.0"
LABEL description="Interpretable Deep-Learning and Ensemble Models for Predicting Multidrug Resistance in K. pneumoniae"
LABEL pipeline.version="1.0"
LABEL pipeline.snakemake=">=7.0.0"