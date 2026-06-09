# UCARE 2026-27: Early-Halting Criteria in LLMs

Investigating principled early-halting criteria in large language models
using the [Pythia model suite](https://github.com/EleutherAI/pythia).

This project examines when task-relevant information becomes linearly
accessible across transformer hidden layers. By comparing factual lookup
tasks against algorithmic reasoning tasks, we aim to design early-halting
triggers grounded in representation geometry (intrinsic dimension,
residual-stream stability) rather than output-space heuristics (softmax
entropy).

Licensed under Apache 2.0.

## Project Structure

```
ucare-26/
├── src/
│   ├── extraction/    # Hidden state extraction from Pythia models
│   ├── probing/       # Linear probes (future)
│   ├── metrics/       # Evaluation metrics (future)
│   └── data/          # Data loading and processing (future)
├── configs/           # Hydra/YAML configuration files
├── scripts/           # SLURM job scripts
├── tests/             # pytest test suite
└── notebooks/         # Jupyter notebooks for analysis
```

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

## Running the Smoke Test

The smoke test loads `pythia-70m-deduped` and verifies hidden state
extraction:

```bash
python -m src.extraction.smoke_test
```

On a SLURM cluster:

```bash
sbatch scripts/run_smoke_test.sh
```

## Running Tests

```bash
pytest
```

Tests use mocked models and do not require GPU or model downloads.

## Configuration

Configuration files use YAML format compatible with Hydra:

- `configs/model.yaml` -- Model selection and precision settings
- `configs/extraction.yaml` -- Extraction pipeline parameters
