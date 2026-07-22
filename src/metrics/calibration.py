"""
Expected calibration error (ECE) for classification probes.

ECE measures whether a probe's predicted confidence tracks its actual
accuracy: if the probe reports 90% confidence across many examples, are
roughly 90% of them actually correct? A well-calibrated probe's
confidence is trustworthy on its own; a poorly calibrated one might be
accurate overall while being systematically over- or under-confident.
Classification-only — there is no notion of predicted-class confidence
for a continuous regression target.
"""

import logging
from typing import Any

import numpy as np

from src.probing.probes import LinearProbe

logger = logging.getLogger(__name__)


def expected_calibration_error(
    probe: LinearProbe,
    X: Any,
    y: Any,
    n_bins: int = 10,
) -> float:
    """Compute the expected calibration error of a classification probe.

    Bins examples by their predicted-class confidence (the maximum
    predicted probability) into *n_bins* equal-width bins spanning
    ``[0, 1]``, then computes the weighted average gap between each
    bin's mean confidence and its empirical accuracy.

    Args:
        probe: A fitted classification ``LinearProbe``.
        X: Feature matrix of shape ``(n_samples, hidden)``.
        y: True labels of shape ``(n_samples,)``.
        n_bins: Number of equal-width confidence bins.

    Returns:
        The expected calibration error, in ``[0, 1]``. Lower values
        indicate better-calibrated confidence.

    Raises:
        ValueError: If *probe* is a regression probe.
    """
    if probe.task_spec.label_type != "classification":
        raise ValueError(
            "expected_calibration_error is only defined for classification "
            "probes"
        )

    y = np.asarray(y)
    probabilities = probe.predict_proba(X)
    confidences = probabilities.max(axis=1)
    predictions = probe.predict(X)
    correct = (predictions == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_samples = len(y)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue

        bin_confidence = float(confidences[in_bin].mean())
        bin_accuracy = float(correct[in_bin].mean())
        ece += (bin_count / n_samples) * abs(bin_accuracy - bin_confidence)

    logger.info(
        "ECE for task=%s: %.4f (n_bins=%d, n_samples=%d)",
        probe.task_spec.name,
        ece,
        n_bins,
        n_samples,
    )

    return float(ece)
