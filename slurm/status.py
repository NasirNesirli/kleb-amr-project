#!/usr/bin/env python3
"""
Slurm job status checker for Snakemake
Checks job status and returns appropriate exit codes
"""

import sys
import subprocess
import time

def get_job_status(job_id):
    """Get job status from Slurm"""
    try:
        cmd = f"sacct -j {job_id} --format=State --noheader --parsable2"
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return "UNKNOWN"
        
        status = result.stdout.strip().split('\n')[0]
        return status
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception:
        return "UNKNOWN"

def main():
    if len(sys.argv) != 2:
        print("Usage: status.py <job_id>", file=sys.stderr)
        sys.exit(1)
    
    job_id = sys.argv[1]
    status = get_job_status(job_id)
    
    # Map Slurm states to Snakemake expected exit codes
    if status in ["COMPLETED"]:
        sys.exit(0)  # Success
    elif status in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]:
        sys.exit(1)  # Still running
    elif status in ["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"]:
        sys.exit(2)  # Failed
    else:
        # Unknown status, assume still running
        sys.exit(1)

if __name__ == "__main__":
    main()