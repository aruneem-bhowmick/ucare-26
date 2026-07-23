"""
Linear probing module.

Provides linear probe classifiers for evaluating the linear
decodability of task-relevant information across transformer layers.
"""

from src.probing.cross_validation import CVResult, run_cross_validated_probe
from src.probing.pipeline import ProbingPipeline
from src.probing.probes import LinearProbe, load_probing_config
from src.probing.selectivity import SelectivityResult, compute_selectivity

__all__ = [
    "CVResult",
    "LinearProbe",
    "ProbingPipeline",
    "SelectivityResult",
    "compute_selectivity",
    "load_probing_config",
    "run_cross_validated_probe",
]
