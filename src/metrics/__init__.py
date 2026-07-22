"""
Evaluation metrics module.

Provides layer-wise probing diagnostics — classification margin,
expected calibration error, and prediction depth — that quantify probe
separability, confidence calibration, and per-example prediction
stability beyond raw probe accuracy.
"""

from src.metrics.calibration import expected_calibration_error
from src.metrics.margins import classification_margin
from src.metrics.prediction_depth import prediction_depth

__all__ = [
    "classification_margin",
    "expected_calibration_error",
    "prediction_depth",
]
