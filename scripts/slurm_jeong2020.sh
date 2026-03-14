#!/bin/bash
#SBATCH --job-name=jeong2020
#SBATCH --array=1-25
#SBATCH --partition=gpu-best,normal-best,parietal,gamma,grace,comete,tribe,tau
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/home/tau/baristim/jeong2020_%A_%a.log

# Jeong2020: Job array — one task per subject (1-25).
# Each task downloads via s5cmd, resamples 2500->1000 Hz, ZIPs.

set -euo pipefail

SUBJECT=${SLURM_ARRAY_TASK_ID}
WORK_DIR="/data/tau/iceberg_1/shared/jeong2020_tmp"
OUTPUT_DIR="/home/tau/baristim/mne_data/jeong2020_zenodo"

echo "=== Jeong2020 subject ${SUBJECT} ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

# Activate conda
source /home/tau/baristim/miniforge3/etc/profile.d/conda.sh
conda activate base

python /home/tau/baristim/moabb-ssvep/scripts/repackage_jeong2020.py \
    --work-dir "$WORK_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --subjects "$SUBJECT" \
    --cleanup

echo "=== Subject ${SUBJECT} done: $(date) ==="
