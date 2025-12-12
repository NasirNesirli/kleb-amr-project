#!/usr/bin/env python3
"""
Process raw metadata file to extract representative drug phenotypes.
Creates train (pre-2023) and test (2023-24) splits.
"""

import pandas as pd
import re
import sys
import subprocess
from pathlib import Path

def parse_ast_phenotypes(ast_string, target_drugs):
    """Parse AST phenotypes string to extract target drug results."""
    results = {}
    if pd.isna(ast_string):
        return {drug: None for drug in target_drugs}
    
    for drug in target_drugs:
        pattern = rf'{drug}=([RSI])'
        match = re.search(pattern, ast_string, re.IGNORECASE)
        if match:
            phenotype = match.group(1).upper()
            # Convert to binary: R=1, S=0, I=exclude
            if phenotype == 'R':
                results[drug] = 1
            elif phenotype == 'S':
                results[drug] = 0
            else:
                results[drug] = None
        else:
            results[drug] = None
    return results

def check_library_layout(run_id):
    """Check if sample is paired-end by querying SRA database."""
    try:
        # Use esearch and efetch to get library layout from SRA
        cmd = f"esearch -db sra -query '{run_id}' | efetch -format runinfo | cut -d',' -f16"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            layout = result.stdout.strip().split('\n')[-1]  # Get last line (skip header)
            if layout.upper() == 'PAIRED':
                return 'paired'
            elif layout.upper() == 'SINGLE':
                return 'single'
        
        # Fallback: try sra-stat if available
        cmd2 = f"sra-stat --quick {run_id} 2>/dev/null | grep 'layout:' || echo 'unknown'"
        result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=30)
        if 'paired' in result2.stdout.lower():
            return 'paired'
        elif 'single' in result2.stdout.lower():
            return 'single'
            
    except Exception as e:
        print(f"Warning: Could not check layout for {run_id}: {e}")
    
    return 'unknown'

def main():
    # Load parameters from snakemake
    metadata_file = snakemake.input.metadata
    train_output = snakemake.output.train
    test_output = snakemake.output.test
    train_cutoff = snakemake.params.train_cutoff
    test_years = snakemake.params.test_years
    antibiotics = snakemake.params.antibiotics
    
    # Read metadata
    df = pd.read_csv(metadata_file, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.replace('#', '')
    
    # Rename columns
    df = df.rename(columns={
        'Run': 'Run',
        'Collection date': 'Collection_date',
        'AST phenotypes': 'AST_phenotypes',
        'Isolate': 'Isolate',
        'Location': 'Location',
        'Isolation source': 'Isolation_source'
    })
    
    # Extract year from collection date
    df['Year'] = pd.to_numeric(df['Collection_date'].astype(str).str[:4], errors='coerce')
    
    # Parse AST phenotypes for target antibiotics
    for drug in antibiotics:
        df[drug] = df['AST_phenotypes'].apply(
            lambda x: parse_ast_phenotypes(x, [drug]).get(drug)
        )
    
    # Select columns
    cols = ['Run', 'Collection_date', 'Year', 'Isolate', 'Location', 
            'Isolation_source'] + antibiotics
    df = df[cols]
    
    # Filter samples with at least one valid phenotype
    df = df.dropna(subset=antibiotics, how='all')
    
    # Check library layout for all samples
    print("Checking library layout for all samples...")
    paired_samples = []
    single_samples = []
    unknown_samples = []
    
    for _, row in df.iterrows():
        run_id = row['Run']
        layout = check_library_layout(run_id)
        print(f"{run_id}: {layout}")
        
        if layout == 'paired':
            paired_samples.append(run_id)
        elif layout == 'single':
            single_samples.append(run_id)
        else:
            unknown_samples.append(run_id)
    
    print(f"\nLibrary layout summary:")
    print(f"Paired-end samples: {len(paired_samples)}")
    print(f"Single-end samples: {len(single_samples)}")
    print(f"Unknown layout: {len(unknown_samples)}")
    
    if single_samples:
        print(f"Excluding single-end and unknown samples: {single_samples} , {unknown_samples}")
        df = df[df['Run'].isin(paired_samples)]
        print(f"Samples after filtering: {len(df)}")
    
    # Split by year
    train_df = df[df['Year'] <= train_cutoff].copy()
    test_df = df[df['Year'].isin(test_years)].copy()
    
    print(f"Training samples (≤{train_cutoff}): {len(train_df)}")
    print(f"Test samples ({test_years}): {len(test_df)}")
    
    # Save processed metadata 
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    print(f"Saved processed training metadata to {train_output}")
    print(f"Saved processed test metadata to {test_output}")

if __name__ == "__main__":
    main()