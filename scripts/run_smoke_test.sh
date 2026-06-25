#!/bin/bash
#SBATCH --job-name=pythia-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# Smoke test and pipeline validation wrapper for SLURM or local execution.
#
# This script can be run directly (bash scripts/run_smoke_test.sh)
# or submitted to a SLURM scheduler (sbatch scripts/run_smoke_test.sh).
# It runs the legacy smoke test (pythia-70m-deduped, dummy sentence) followed
# by the full pipeline validation on Pythia-70M and 160M with SST-2 data.
#
# Usage:
#   Local:  bash scripts/run_smoke_test.sh
#   SLURM:  sbatch scripts/run_smoke_test.sh

set -euo pipefail

echo "=== Pythia Smoke Test ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(python --version)"

# Create logs directory if it doesn't exist (needed for SLURM output)
mkdir -p logs

python -m src.extraction.smoke_test

echo "=== Smoke test completed successfully ==="
