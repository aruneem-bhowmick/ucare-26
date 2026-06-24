"""
Data handling module.

Provides task specifications, data loaders for standard HuggingFace
datasets (SST-2, MRPC, LAMA/T-REx), synthetic generators for
algorithmic and structured-knowledge tasks (Dyck-k, modular
arithmetic, periodic table), and a shuffled-label control generator
for probe selectivity analysis.
"""

from src.data.controls import generate_shuffled_labels
from src.data.loaders import load_lama_trex, load_mrpc, load_sst2
from src.data.synthetic import (
    generate_dyck,
    generate_modular_arithmetic,
    generate_periodic_table,
)
from src.data.tasks import TaskSpec, get_task_spec, load_task_registry

__all__ = [
    "TaskSpec",
    "load_task_registry",
    "get_task_spec",
    "load_sst2",
    "load_mrpc",
    "load_lama_trex",
    "generate_dyck",
    "generate_modular_arithmetic",
    "generate_periodic_table",
    "generate_shuffled_labels",
]
