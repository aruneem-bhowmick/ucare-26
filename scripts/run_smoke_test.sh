#!/bin/bash
#SBATCH --job-name=pythia-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# Smoke test wrapper for SLURM or local execution.
#
# This script can be run directly (bash scripts/run_smoke_test.sh)
# or submitted to a SLURM scheduler (sbatch scripts/run_smoke_test.sh).
# It loads the pythia-70m-deduped model and validates hidden state
# extraction on a single GPU.
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
