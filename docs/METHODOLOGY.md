# Methodology Documentation

## Overview

This document details the methodological approach for the AMR prediction pipeline, including statistical considerations, model architectures, and validation strategies.

## Data Processing

### Temporal Splitting Strategy
- **Training Data**: Samples collected ≤ 2022
- **Test Data**: Samples from 2023-2024
- **Rationale**: Mimics real-world deployment scenario
- **Limitation**: Insufficient 2025 data available

### Quality Control Pipeline
1. **Read-level QC**: FastQC + MultiQC reporting
2. **Trimming**: Fastp with quality score ≥ 20
3. **Assembly**: SPAdes with coverage downsampling to 100×
4. **Contamination Screening**: Kraken2 + Bracken
5. **Assembly QC**: QUAST + CheckM

### Feature Engineering

#### AMR Gene Detection
- **Tool**: NCBI AMRFinderPlus
- **Database**: NCBI Pathogen Detection reference
- **Processing**: Binary presence/absence matrix

#### SNP Calling
- **Reference**: K. pneumoniae HS11286 (GCF_000240185.1)
- **Pipeline**: BWA alignment → GATK variant calling
- **Filters**: QUAL ≥ 30, DP ≥ 10, AF ≥ 0.9

#### K-mer Features
- **K-mer size**: 11 nucleotides
- **Tool**: Jellyfish for counting
- **Normalization**: Log10 transform + standardization

## Model Architectures

### Tree-Based Models

#### XGBoost Configuration
```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=neg_count/pos_count,  # Class balancing
    random_state=42
)
```

#### LightGBM Configuration
```python
LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    scale_pos_weight=neg_count/pos_count,  # Consistent with XGBoost
    random_state=42
)
```

### Deep Learning Models

#### 1D-CNN Architecture
```
Input: K-mer features (n_features,)
Conv1D(filters=32, kernel_size=3, activation='relu')
BatchNorm1D()
Dropout(0.3)
Conv1D(filters=64, kernel_size=3, activation='relu')
BatchNorm1D()
GlobalMaxPooling1D()
Dense(128, activation='relu')
Dropout(0.5)
Dense(2, activation='softmax')  # With class weights
```

#### Sequence CNN Architecture
```
Input: One-hot encoded sequences (4, seq_length)
Conv1D(filters=32, kernel_size=7, activation='relu')
MaxPooling1D(pool_size=3)
Conv1D(filters=64, kernel_size=5, activation='relu')
MaxPooling1D(pool_size=3)
Flatten()
Dense(128, activation='relu')
Dropout(0.5)
Dense(2, activation='softmax')
```

#### DNABERT-2 Fine-tuning
```python
# Base model: zhihan1996/DNABERT-2-117M from HuggingFace
class DNABERT2Classifier(nn.Module):
    def __init__(self):
        self.dnabert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 2)  # 768 = DNABERT hidden size
        )
    
# Training: 250bp sequences with class-weighted CrossEntropyLoss
```

## Cross-Validation Strategy

### Geographic-Temporal Grouping
```python
def create_location_year_groups(metadata):
    """Create groups based on location + collection year"""
    groups = metadata['Location'] + '_' + metadata['Year'].astype(str)
    return groups

class GeographicTemporalKFold:
    """Ensures samples from same location-year stay together"""
    def split(self, X, y, groups):
        # Group assignment prevents strain leakage
        # Falls back to StratifiedKFold if insufficient groups
```

### Rationale
- **Prevents Data Leakage**: Related strains don't appear in both train/validation
- **Epidemiological Realism**: Accounts for geographic clustering and temporal trends
- **Literature Support**: Recommended for genomic ML (Moradigaravand et al., 2018)

## Class Imbalance Handling

### Standardized Approach
```python
def compute_balanced_weights(y_train):
    """Consistent class weighting across all models"""
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    
    # Tree models
    scale_pos_weight = neg_count / pos_count
    
    # Deep models  
    class_weights = torch.tensor([1.0, neg_count/pos_count])
    
    return scale_pos_weight, class_weights
```

### Model-Specific Implementation
- **XGBoost/LightGBM**: `scale_pos_weight` parameter
- **PyTorch Models**: `weight` parameter in `CrossEntropyLoss`
- **No Sampling**: Avoids `WeightedRandomSampler` to prevent double-balancing

## Hyperparameter Optimization

### Grid Search Strategy
```python
param_grids = {
    'xgboost': {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'num_leaves': [31, 63, 127]
    }
}
```

### Deep Learning Training
- **Optimizer**: Adam with default parameters
- **Learning Rate**: 0.001 (tree models), 0.0001 (transformers)
- **Batch Size**: 32 (CNN), 16 (DNABERT)
- **Early Stopping**: Based on validation F1 score
- **Epochs**: 100 (CNN), 50 (DNABERT)

## Statistical Testing Framework

### DeLong's Test for ROC Comparison
```python
def delong_test(y_true, y_scores1, y_scores2):
    """Compare AUC between two models (DeLong et al. 1988)"""
    # Implementation follows original algorithm
    # Returns z-statistic and p-value
```

### Multiple Comparison Correction
- **Method**: Bonferroni correction
- **Alpha Level**: 0.01 (conservative)
- **Rationale**: Controls family-wise error rate across pairwise comparisons

### Bootstrap Confidence Intervals
```python
def bootstrap_ci(metric_func, y_true, y_pred, n_bootstrap=1000):
    """Non-parametric confidence intervals"""
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    # 95% CI
    return np.percentile(bootstrap_scores, [2.5, 97.5])
```

## Feature Selection Pipeline

### Multi-Stage Approach
1. **Variance Filtering**: Remove near-constant features (threshold=0.001)
2. **Sparsity Filtering**: Remove >95% sparse features (preserve AMR genes)
3. **Statistical Selection**: Chi-square + mutual information
4. **Clinical Relevance**: Preserve known resistance genes regardless of statistics

### Batch Effect Assessment
```python
def assess_batch_effects(data, batch_vars=['Year', 'Location']):
    """PCA-based batch effect detection"""
    # Explained variance by batch variables
    # Recommend ComBat correction if >50% variance explained
```

## Interpretability Analysis

### Feature Importance Aggregation
```python
def consensus_importance(model_importances):
    """Aggregate feature importance across models"""
    # Rank-based aggregation to handle different scales
    # Weight by model performance (F1 score)
    consensus_score = sum(rank * weight for rank, weight in zip(ranks, weights))
```

### Motif Analysis
- **CNN Filters**: Extract sequence patterns from learned filters
- **DNABERT Attention**: High-attention regions → sequence motifs
- **K-mer Clustering**: Find common subsequences in important k-mers
- **Cross-Model Consensus**: Motifs appearing across multiple model types

### Clinical Validation
- **AMR Gene Database**: Compare against NCBI AMRFinderPlus annotations
- **Literature Mining**: Known resistance mechanisms for each antibiotic
- **Functional Categories**: β-lactamases, efflux pumps, target modifications

## Model Evaluation

### Primary Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Average of per-class accuracies
- **AUC-ROC**: Area under receiver operating characteristic curve

### Success Criteria (Project Proposal)
- **Primary**: F1 ≥ 0.85 per antibiotic
- **Secondary**: Balanced accuracy ≥ 0.85
- **Discrepancy Analysis**: If balanced accuracy meets threshold but F1 doesn't

### Cross-Model Comparison
```python
def statistical_model_comparison(results):
    """Comprehensive model comparison with statistical tests"""
    # DeLong's test for AUC comparison
    # Bootstrap CI for all metrics
    # Friedman test for multiple model comparison
    # Effect size calculation (Cohen's d)
```

## Computational Considerations

### Hardware Requirements
- **Memory**: 16 GB RAM (minimum 8 GB)
- **CPU**: 8 cores recommended
- **GPU**: Optional for deep learning (CUDA-compatible)
- **Storage**: 50 GB for full pipeline

### Performance Optimization
- **Parallel Processing**: Snakemake job parallelization
- **Memory Management**: Chunked processing for large feature matrices
- **Caching**: Intermediate results saved to avoid recomputation

## Reproducibility Measures

### Random State Management
```python
# Consistent random states across all scripts
RANDOM_STATE = 42

# Set seeds for all libraries
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
```

### Version Control
- **Conda Environments**: Pinned package versions
- **Model Checkpoints**: Saved trained models for replication
- **Configuration Files**: All parameters in version-controlled YAML

### Data Lineage
- **Snakemake**: Tracks all file dependencies
- **Metadata**: Preserved throughout pipeline
- **Provenance**: Each output file linked to input and processing steps

## Limitations and Future Work

### Current Limitations
1. **Small Dataset**: Test dataset (~57 samples) vs. proposed 1000
2. **Temporal Coverage**: Missing 2025 data for validation
3. **Geographic Bias**: Overrepresentation of certain regions
4. **Resistance Mechanisms**: Limited to gene presence/absence

### Future Enhancements
1. **Larger Datasets**: Scale to full NCBI Pathogen Detection database
2. **Multi-Drug Resistance**: Predict resistance profiles across drug classes
3. **Ensemble Methods**: Weighted combination of individual models
4. **Real-Time Prediction**: API for clinical integration
5. **Causal Analysis**: Identify resistance evolution pathways