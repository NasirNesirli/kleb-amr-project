"""
Step 04: Genome Assembly with SPAdes
Downsample to 100x, assemble, rename and keep only final assembly
Run independently: snakemake --use-conda --cores 8 -s rules/04_assembly.smk assembly_all
"""

import pandas as pd

configfile: "config/config.yaml"

rule estimate_genome_size:
    output:
        "data/reference/genome_size.txt"
    shell:
        """
        echo "5500000" > {output}
        """

rule downsample_reads:
    input:
        r1="data/processed/{sample}_1.fastq.gz",
        r2="data/processed/{sample}_2.fastq.gz",
        genome_size="data/reference/genome_size.txt"
    output:
        r1=temp("data/processed/{sample}_ds_1.fastq.gz"),
        r2=temp("data/processed/{sample}_ds_2.fastq.gz")
    params:
        target_cov=config["qc"]["target_coverage"]
    conda:
        "../envs/assembly.yaml"
    log:
        "logs/04_assembly/downsample_{sample}.log"
    shell:
        """
        python scripts/04_run_assembly.py downsample \
            --r1 {input.r1} --r2 {input.r2} \
            --out_r1 {output.r1} --out_r2 {output.r2} \
            --genome_size $(cat {input.genome_size}) \
            --target_cov {params.target_cov} 2> {log}
        """

rule spades_assembly:
    input:
        r1="data/processed/{sample}_ds_1.fastq.gz",
        r2="data/processed/{sample}_ds_2.fastq.gz"
    output:
        assembly="data/assemblies/{sample}_assembled.fasta"
    params:
        outdir=temp(directory("data/assemblies/{sample}_spades")),
        memory=config["resources"]["memory_gb"]
    conda:
        "../envs/assembly.yaml"
    threads: config["resources"]["threads"]
    log:
        "logs/04_assembly/spades_{sample}.log"
    shell:
        """
        spades.py -1 {input.r1} -2 {input.r2} \
            -o {params.outdir} \
            -t {threads} -m {params.memory} \
            --isolate 2> {log}
        
        cp {params.outdir}/contigs.fasta {output.assembly}
        rm -rf {params.outdir}
        
        # Immediately delete downsampled files to save disk space
        rm -f {input.r1} {input.r2}
        """

rule cleanup_raw_after_assembly:
    input:
        assembly="data/assemblies/{sample}_assembled.fasta"
    output:
        touch("data/assemblies/.{sample}_cleanup_done")
    shell:
        """
        rm -f data/raw/{wildcards.sample}_*.fastq.gz
        """

def get_all_assemblies(wildcards):
    train_df = pd.read_csv("results/features/metadata_train_processed.csv")
    test_df = pd.read_csv("results/features/metadata_test_processed.csv")
    samples = list(train_df["Run"]) + list(test_df["Run"])
    return expand("data/assemblies/{sample}_assembled.fasta", sample=samples)

rule assembly_all:
    input:
        get_all_assemblies