"""
K-fold cross-validation harness with a regularization sweep.

Wraps ``LinearProbe`` with stratified k-fold cross-validation for
classification tasks and plain k-fold for regression tasks (Lei &
Cooper, 2025), sweeping a regularization grid and selecting the value
with the best mean held-out score.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from src.data.tasks import TaskSpec
from src.probing.probes import LinearProbe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVResult:
    """Result of a cross-validated regularization sweep for one probe.

    Attributes:
        best_regularization: The regularization value (from the swept
            grid) with the best mean held-out fold score.
        fold_scores: Per-fold held-out scores at
            ``best_regularization``, one per cross-validation fold.
        mean_score: Mean of ``fold_scores``.
        std_score: Population standard deviation of ``fold_scores``.
        probe: A ``LinearProbe`` fit on the entire ``(X, y)`` at
            ``best_regularization``. Downstream margin, calibration,
            and prediction-depth analyses use this probe directly
            rather than re-running cross-validation.
    """

    best_regularization: float
    fold_scores: list[float]
    mean_score: float
    std_score: float
    probe: LinearProbe


def _build_splitter(
    task_spec: TaskSpec, folds: int, seed: int
) -> StratifiedKFold | KFold:
    """Select the fold splitter appropriate for the task's label type.

    Args:
        task_spec: Task specification whose ``label_type`` determines
            the splitter.
        folds: Number of cross-validation folds.
        seed: Random seed for fold shuffling.

    Returns:
        A ``StratifiedKFold`` for classification tasks (so class
        proportions are preserved per fold) or a plain ``KFold`` for
        regression tasks (stratification is undefined for continuous
        targets).
    """
    if task_spec.label_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def run_cross_validated_probe(
    X: Any,
    y: Any,
    task_spec: TaskSpec,
    config: dict[str, Any],
) -> CVResult:
    """Cross-validate a linear probe over a regularization grid.

    Splits ``(X, y)`` into ``config["cv"]["folds"]`` folds (stratified
    for classification, plain for regression), fits a fresh
    ``LinearProbe`` per fold for every value in
    ``config["probe"]["regularization_grid"]`` on identical splits,
    and selects the regularization value with the best mean held-out
    score. A final probe is then refit on the full dataset at that
    regularization value.

    Only the ``probe`` and ``cv`` sections of *config* are read;
    other top-level sections (``models``, ``tasks``, ``output_dir``)
    are ignored, matching the section-scoping convention used
    elsewhere in this package.

    Args:
        X: Feature matrix of shape ``(n_samples, hidden)``. May be a
            ``torch.Tensor`` (including float16) or array-like — fold
            indexing works identically for both, and dtype handling
            is delegated to ``LinearProbe``.
        y: Target labels or values of shape ``(n_samples,)``.
        task_spec: Task specification determining classification vs.
            regression dispatch.
        config: Parsed probing configuration (as returned by
            ``load_probing_config``).

    Returns:
        A ``CVResult`` describing the best regularization value, its
        per-fold scores, their mean/std, and a probe refit on the
        full dataset.
    """
    probe_cfg = config["probe"]
    cv_cfg = config["cv"]
    folds = cv_cfg["folds"]
    seed = cv_cfg["seed"]
    regularization_grid = probe_cfg["regularization_grid"]

    y_arr = np.asarray(y)

    splitter = _build_splitter(task_spec, folds, seed)
    splits = list(splitter.split(X, y_arr))

    def _make_probe(regularization: float) -> LinearProbe:
        return LinearProbe(
            task_spec,
            regularization=regularization,
            standardize=probe_cfg["standardize"],
            class_weight=probe_cfg.get("class_weight"),
            classification_algorithm=probe_cfg["classification"],
            regression_algorithm=probe_cfg["regression"],
            seed=seed,
        )

    best_regularization: float | None = None
    best_mean_score = float("-inf")
    best_fold_scores: list[float] = []

    for regularization in regularization_grid:
        fold_scores: list[float] = []
        for train_idx, test_idx in splits:
            probe = _make_probe(regularization)
            probe.fit(X[train_idx], y_arr[train_idx])
            fold_scores.append(probe.score(X[test_idx], y_arr[test_idx]))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_regularization = regularization
            best_fold_scores = fold_scores

    assert best_regularization is not None  # regularization_grid is non-empty

    final_probe = _make_probe(best_regularization)
    final_probe.fit(X, y_arr)

    logger.info(
        "Cross-validated probe for task=%s: best_regularization=%s, "
        "mean_score=%.4f, std_score=%.4f",
        task_spec.name,
        best_regularization,
        best_mean_score,
        float(np.std(best_fold_scores)),
    )

    return CVResult(
        best_regularization=best_regularization,
        fold_scores=best_fold_scores,
        mean_score=best_mean_score,
        std_score=float(np.std(best_fold_scores)),
        probe=final_probe,
    )
