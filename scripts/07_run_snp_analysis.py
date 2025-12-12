#!/usr/bin/env python3
"""
Combine filtered VCF files into SNP presence/absence matrix.
"""

import pandas as pd
from pathlib import Path
import re

def parse_vcf(vcf_file):
    """Parse VCF file and extract SNP positions."""
    snps = []
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                chrom, pos, _, ref, alt = parts[:5]
                snp_id = f"{chrom}_{pos}_{ref}>{alt}"
                snps.append(snp_id)
    return set(snps)

def main():
    input_files = snakemake.input
    output_file = snakemake.output[0]
    
    all_snps = set()
    sample_snps = {}
    
    # Parse all VCF files
    for f in input_files:
        sample_id = Path(f).stem.replace('.filtered', '')
        snps = parse_vcf(f)
        all_snps.update(snps)
        sample_snps[sample_id] = snps
    
    # Create presence/absence matrix
    all_snps = sorted(all_snps)
    matrix_data = []
    
    for sample_id, snps in sample_snps.items():
        row = {'sample_id': sample_id}
        for snp in all_snps:
            row[f'snp_{snp}'] = 1 if snp in snps else 0
        matrix_data.append(row)
    
    matrix_df = pd.DataFrame(matrix_data)
    matrix_df.to_csv(output_file, index=False)
    
    print(f"Combined {len(sample_snps)} samples with {len(all_snps)} unique SNPs")

if __name__ == "__main__":
    main()