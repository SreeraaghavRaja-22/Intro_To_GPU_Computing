#!/bin/bash

# Check if JobID was actually provided
if [ -z "$1" ]; then 
    echo "Error: missing Slurm Job ID."
    echo "Usage: $0 <job_id>"
fi

JOB_ID="$1"

srun --jobid="$JOB_ID" --pty /bin/bash