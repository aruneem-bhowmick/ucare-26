"""
Per-example prediction-depth estimator from layer-wise probe predictions.

Prediction depth (Belrose et al., 2023) is the earliest layer at which
an example's top-1 prediction settles into whatever it remains for the
rest of the network's depth — the start of the longest stable suffix
of predictions ending at the final layer. It is a per-example
difficulty signal: an easy example's prediction depth is shallow (the
representation already carries the answer early), while a hard
example's prediction depth sits near the final layer (the prediction
keeps changing as computation proceeds).
"""

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def prediction_depth(
    per_layer_predictions: Sequence[Any],
    final_prediction: Any | None = None,
) -> int:
    """Find the earliest layer starting the longest stable prediction suffix.

    Scans *per_layer_predictions* (indexed ``0`` through ``L``, shallow
    to deep) from the last layer backward, and returns the earliest
    layer index ``d`` such that every prediction from layer ``d``
    through layer ``L`` equals *final_prediction*.

    Args:
        per_layer_predictions: Top-1 predictions for a single example,
            one per layer, ordered from the shallowest (index 0) to
            the deepest (index ``L``) layer.
        final_prediction: The prediction to treat as the example's
            settled answer. Defaults to the last element of
            *per_layer_predictions* (the deepest layer's prediction).

    Returns:
        The earliest layer index at which the prediction stabilizes to
        *final_prediction* and never changes again through the deepest
        layer. If no such stability exists (the deepest layer's own
        prediction does not equal *final_prediction*, or the sequence
        has length 1), the deepest layer index is returned.

    Raises:
        ValueError: If *per_layer_predictions* is empty.
    """
    predictions = list(per_layer_predictions)
    if not predictions:
        raise ValueError("per_layer_predictions must contain at least one layer")

    target = final_prediction if final_prediction is not None else predictions[-1]

    depth = len(predictions) - 1
    for layer in range(len(predictions) - 1, -1, -1):
        if predictions[layer] != target:
            break
        depth = layer

    return depth
