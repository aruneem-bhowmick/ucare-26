"""
Shuffled-label control generator for probe selectivity analysis.

A probe trained on shuffled labels should perform at chance if it is
genuinely reading task-relevant structure from the representation
rather than memorizing the training set.  Selectivity is defined as
task accuracy minus control accuracy (Hewitt & Liang, 2019); a high
selectivity score indicates the probe is leveraging learned features,
not overfitting to the label space.

This module provides ``generate_shuffled_labels``, which takes any
``Dataset`` produced by the loaders or synthetic generators and
returns a copy with the label column randomly permuted.
"""

import logging
import random

from datasets import Dataset

logger = logging.getLogger(__name__)


def generate_shuffled_labels(
    dataset: Dataset,
    label_column: str = "label",
    seed: int = 42,
) -> Dataset:
    """Return a copy of *dataset* with the label column randomly permuted.

    All other columns (``text``, ``example_id``, metadata) are left
    unchanged so that the control dataset is identical to the original
    in every respect except the label–feature association.

    Works for both integer classification labels and numeric regression
    targets — the permutation preserves the marginal label distribution
    exactly.

    Args:
        dataset: A ``Dataset`` with at least a *label_column*.
        label_column: Name of the column to shuffle. Defaults to
            ``"label"``.
        seed: Random seed for reproducible permutation.

    Returns:
        A new ``Dataset`` with the same columns and length, but with
        *label_column* values randomly permuted.

    Raises:
        ValueError: If *label_column* is not present in *dataset*.
    """
    if label_column not in dataset.column_names:
        raise ValueError(
            f"Column {label_column!r} not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    labels = list(dataset[label_column])
    rng = random.Random(seed)
    rng.shuffle(labels)

    shuffled = dataset.remove_columns([label_column])
    shuffled = shuffled.add_column(label_column, labels)

    logger.info(
        "Generated shuffled-label control: %d examples, column=%s",
        len(shuffled),
        label_column,
    )
    return shuffled
