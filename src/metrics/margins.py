"""
Classification margin computation for linear probes.

The margin is the signed distance between an example and a probe's
decision boundary: how far past the boundary (and on which side) an
example lands, rather than merely whether its prediction is right or
wrong. It summarizes separability beyond raw accuracy — a probe can be
100% accurate with a razor-thin margin on every example, or 100%
accurate with wide margins throughout, and only the margin
distinguishes the two.
"""

import logging
from typing import Any

import numpy as np

from src.probing.probes import LinearProbe

logger = logging.getLogger(__name__)


def classification_margin(probe: LinearProbe, X: Any, y: Any) -> np.ndarray:
    """Compute the signed margin of each example to the decision boundary.

    For binary classification, this is the raw ``decision_function``
    output, signed so that a positive value means the example lands on
    the correct side of the boundary for its true label and a negative
    value means it lands on the wrong side. For multi-class problems,
    this generalizes to a one-vs-rest margin: the true class's decision
    score minus the highest decision score among all other classes,
    which is positive exactly when the true class would win a
    one-vs-rest vote against every other class.

    Args:
        probe: A fitted classification ``LinearProbe``.
        X: Feature matrix of shape ``(n_samples, hidden)``.
        y: True labels of shape ``(n_samples,)``.

    Returns:
        A 1-D array of per-example signed margins.

    Raises:
        AttributeError: If *probe* is a regression probe (propagated
            from ``LinearProbe.decision_function``).
    """
    y = np.asarray(y)
    scores = probe.decision_function(X)
    classes = probe.pipeline.named_steps["estimator"].classes_

    if scores.ndim == 1:
        sign = np.where(y == classes[1], 1.0, -1.0)
        return sign * scores

    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    true_idx = np.array([class_to_idx[label] for label in y])

    row_idx = np.arange(scores.shape[0])
    true_scores = scores[row_idx, true_idx]

    other_scores = scores.copy()
    other_scores[row_idx, true_idx] = -np.inf
    other_max = other_scores.max(axis=1)

    return true_scores - other_max
