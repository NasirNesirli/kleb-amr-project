"""
Step 06: AMR Gene Detection with AMRFinderPlus
Runs on assemblies to detect AMR genes (presence/absence)
Run independently: snakemake --use-conda --cores 8 -s rules/06_amr_analysis.smk
"""

import pandas as pd

configfile: "config/config.yaml"

rule update_amrfinder_db:
    output:
        touch("data/reference/.amrfinder_db_updated")
    conda:
        "../envs/amr_analysis.yaml"
    log:
        "logs/06_amr_analysis/update_db.log"
    shell:
        """
        amrfinder -u 2> {log}
        """

rule run_amrfinder:
    input:
        assembly="data/assemblies/{sample}_assembled.fasta",
        db_ready="data/reference/.amrfinder_db_updated"
    output:
        "results/amr/{sample}_amrfinder.tsv"
    conda:
        "../envs/amr_analysis.yaml"
    threads: 4
    log:
        "logs/06_amr_analysis/amrfinder_{sample}.log"
    shell:
        """
        amrfinder -n {input.assembly} \
            --organism Klebsiella_pneumoniae \
            --plus \
            --threads {threads} \
            -o {output} 2> {log}
        """

def get_all_amrfinder_outputs(wildcards):
    train_df = pd.read_csv("results/features/metadata_train_processed.csv")
    test_df = pd.read_csv("results/features/metadata_test_processed.csv")
    samples = list(train_df["Run"]) + list(test_df["Run"])
    return expand("results/amr/{sample}_amrfinder.tsv", sample=samples)

rule combine_amrfinder:
    input:
        get_all_amrfinder_outputs
    output:
        "results/amr/combined_amrfinder.csv"
    conda:
        "../envs/amr_analysis.yaml"
    log:
        "logs/06_amr_analysis/combine.log"
    script:
        "../scripts/06_run_amrfinder.py"

rule amr_analysis_all:
    input:
        "results/amr/combined_amrfinder.csv"