#!/bin/bash
#SBATCH --job-name=jeong2020
#SBATCH --partition=gpu-best,normal-best,parietal,gamma,grace,comete,tribe,tau
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/tau/baristim/jeong2020_%j.log

# Jeong2020: Download per-subject files via s5cmd from Wasabi S3,
# resample 2500->1000 Hz, keep ALL channels (EEG + EMG + EOG),
# both MI and realMove. Expected output: ~50 GB in 25 per-subject ZIPs.

set -euo pipefail

WORK_DIR="/data/tau/iceberg_1/shared/jeong2020_tmp"
OUTPUT_DIR="/home/tau/baristim/mne_data/jeong2020_zenodo"

echo "=== Jeong2020 repackage pipeline ==="
echo "Start: $(date)"
echo "Work dir: $WORK_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Disk space (work):"
df -h /data/tau/iceberg_1/
echo "Disk space (output):"
df -h /home/tau/

# Activate conda
source /home/tau/baristim/miniforge3/etc/profile.d/conda.sh
conda activate base

python /home/tau/baristim/moabb-ssvep/scripts/repackage_jeong2020.py \
    --work-dir "$WORK_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --cleanup

echo "=== Done: $(date) ==="
