"""
Control-probe selectivity computation using a shuffled-label baseline.

Selectivity (Hewitt & Liang, 2019) is the gap between a probe trained
on real labels and a matched probe trained on randomly shuffled
labels under the same cross-validation protocol: ``task_score -
control_score``. A probe that is merely memorizing rather than
reading task-relevant structure from the representation will score
similarly on both, giving a selectivity near zero; a probe reading
genuine structure will substantially outperform its shuffled-label
control.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import Dataset

from src.data.controls import generate_shuffled_labels
from src.data.tasks import TaskSpec
from src.probing.cross_validation import CVResult, run_cross_validated_probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectivityResult:
    """Result of a task-probe-vs-control-probe selectivity comparison.

    Attributes:
        task_result: Cross-validation result for the probe trained on
            the real labels.
        control_result: Cross-validation result for the matched probe
            trained on shuffled labels.
        selectivity: ``task_result.mean_score - control_result.mean_score``.
    """

    task_result: CVResult
    control_result: CVResult
    selectivity: float

    @property
    def task_score(self) -> float:
        """Mean cross-validated score of the task probe."""
        return self.task_result.mean_score

    @property
    def control_score(self) -> float:
        """Mean cross-validated score of the shuffled-label control probe."""
        return self.control_result.mean_score


def compute_selectivity(
    X: Any,
    y: Any,
    task_spec: TaskSpec,
    config: dict[str, Any],
) -> SelectivityResult:
    """Compute probe selectivity against a shuffled-label control.

    Trains a cross-validated task probe on the real labels and a
    matched control probe on labels produced by
    ``generate_shuffled_labels`` under the same seed and the same
    cross-validation protocol, then reports the score gap between
    them.

    Note that the task and control cross-validation runs are not
    guaranteed to see identical fold partitions: stratified splitting
    depends on the label array being stratified against, so real vs.
    shuffled labels can produce different (but equally valid) fold
    memberships under the same seed. This is expected, not a data
    leak.

    Args:
        X: Feature matrix of shape ``(n_samples, hidden)``.
        y: Target labels or values of shape ``(n_samples,)``.
        task_spec: Task specification determining classification vs.
            regression dispatch.
        config: Parsed probing configuration (as returned by
            ``load_probing_config``).

    Returns:
        A ``SelectivityResult`` with both cross-validation results and
        the selectivity score.
    """
    task_result = run_cross_validated_probe(X, y, task_spec, config)

    seed = config["cv"]["seed"]
    # example_id is a placeholder column: generate_shuffled_labels removes
    # and re-adds label_column, which collapses a Dataset's row count to
    # zero if label_column is its only column.
    label_dataset = Dataset.from_dict(
        {"label": list(y), "example_id": list(range(len(y)))}
    )
    shuffled_dataset = generate_shuffled_labels(
        label_dataset, label_column="label", seed=seed
    )
    y_control = np.asarray(shuffled_dataset["label"])

    control_result = run_cross_validated_probe(X, y_control, task_spec, config)

    selectivity = task_result.mean_score - control_result.mean_score

    logger.info(
        "Selectivity for task=%s: task_score=%.4f, control_score=%.4f, "
        "selectivity=%.4f",
        task_spec.name,
        task_result.mean_score,
        control_result.mean_score,
        selectivity,
    )

    return SelectivityResult(
        task_result=task_result,
        control_result=control_result,
        selectivity=selectivity,
    )
