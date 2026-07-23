"""
Linear probing module.

Provides linear probe classifiers for evaluating the linear
decodability of task-relevant information across transformer layers.
"""

from src.probing.cross_validation import CVResult, run_cross_validated_probe
from src.probing.probes import LinearProbe, load_probing_config
from src.probing.selectivity import SelectivityResult, compute_selectivity

__all__ = [
    "CVResult",
    "LinearProbe",
    "SelectivityResult",
    "compute_selectivity",
    "load_probing_config",
    "run_cross_validated_probe",
]

# ProbingPipeline is intentionally not re-exported here: it imports
# from src.metrics, and src.metrics.calibration/margins import from
# src.probing.probes, so pulling it into this package's own __init__
# would make src.probing and src.metrics import-order-dependent on
# each other. Import it directly: from src.probing.pipeline import
# ProbingPipeline.
