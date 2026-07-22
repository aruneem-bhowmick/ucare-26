"""
Evaluation metrics module.

Provides layer-wise probing diagnostics — classification margin and
expected calibration error — that quantify probe separability and
confidence calibration beyond raw probe accuracy.
"""

from src.metrics.calibration import expected_calibration_error
from src.metrics.margins import classification_margin

__all__ = [
    "classification_margin",
    "expected_calibration_error",
]
