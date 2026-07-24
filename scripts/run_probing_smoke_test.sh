#!/bin/bash
#SBATCH --job-name=probing-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/probing_smoke_test_%j.out
#SBATCH --error=logs/probing_smoke_test_%j.err

# End-to-end probing pipeline validation wrapper for SLURM or local execution.
#
# This script can be run directly (bash scripts/run_probing_smoke_test.sh)
# or submitted to a SLURM scheduler (sbatch scripts/run_probing_smoke_test.sh).
# It runs the full extract -> probe -> plot pipeline on Pythia-160M against
# SST-2 and a LAMA/T-REx subset.
#
# Usage:
#   Local:  bash scripts/run_probing_smoke_test.sh
#   SLURM:  sbatch scripts/run_probing_smoke_test.sh

set -euo pipefail

echo "=== Probing Pipeline Smoke Test ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(python --version)"

# Create logs directory if it doesn't exist (needed for SLURM output)
mkdir -p logs

python -m src.probing.smoke_test

echo "=== Probing pipeline smoke test completed successfully ==="
