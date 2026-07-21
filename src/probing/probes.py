"""
Linear probe wrapper for classification and regression tasks.

Provides a strictly linear, low-capacity probe (per Hewitt & Liang, 2019:
a high-capacity probe measures what the probe can learn, not what the
representation already encodes) that dispatches to a classification or
regression estimator based on a ``TaskSpec``'s ``label_type``.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from src.data.tasks import TaskSpec

logger = logging.getLogger(__name__)

# Path to the default probing config relative to the project root.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "probing.yaml"

# Allowed classification estimators. Deliberately linear-only: this
# registry is what enforces the "no kernel or hidden-layer options
# exposed" guarantee, since LinearProbe only ever instantiates a class
# looked up here.
_CLASSIFICATION_ESTIMATORS: dict[str, type] = {
    "logistic_regression": LogisticRegression,
}

# Allowed regression estimators (linear_svr uses a linear kernel, not an
# arbitrary one).
_REGRESSION_ESTIMATORS: dict[str, type] = {
    "ridge": Ridge,
    "linear_svr": LinearSVR,
}


def load_probing_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the full probing configuration from YAML.

    Parses ``configs/probing.yaml`` (or the path provided) and returns
    the raw parsed dict, unscoped. Different top-level sections
    (``probe``, ``cv``, ``models``, ``tasks``, ``output_dir``) are
    consumed by different modules: this module only reads ``probe``,
    while the cross-validation harness, selectivity analysis, and
    per-layer pipeline read the others.

    Args:
        config_path: Path to the YAML config file. Defaults to
            ``configs/probing.yaml`` relative to the project root.

    Returns:
        The full parsed configuration as a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info("Loaded probing config from %s", path)
    return config


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor or array-like to a float32/64 numpy array.

    Handles the fp16 safetensors representations produced by
    ``src.extraction.cache.load_representations`` by upcasting to
    float32, since sklearn estimators are not reliably fast or precise
    on float16 input.

    Args:
        x: A ``torch.Tensor`` or anything ``numpy.asarray`` accepts.

    Returns:
        A numpy array with a float32 or float64 dtype (non-float
        dtypes, e.g. integer labels, are passed through unchanged).
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32).numpy()

    array = np.asarray(x)
    if array.dtype == np.float16:
        array = array.astype(np.float32)
    return array


class LinearProbe:
    """Strictly linear probe for classification or regression tasks.

    Dispatches to ``sklearn.linear_model.LogisticRegression`` for
    classification tasks and ``sklearn.linear_model.Ridge`` (or
    ``sklearn.svm.LinearSVR``) for regression tasks, keyed off
    ``task_spec.label_type``. No kernel or hidden-layer options are
    exposed anywhere in this class, by construction.

    Args:
        task_spec: The task specification whose ``label_type``
            (``"classification"`` or ``"regression"``) determines
            which estimator family is used.
        regularization: Regularization strength. Interpreted as ``C``
            (inverse strength) for ``logistic_regression`` and
            ``linear_svr``, and as ``alpha`` (direct strength) for
            ``ridge``.
        standardize: Whether to standardize features (zero mean, unit
            variance) before fitting. When ``True``, the scaler is
            fit only on the data passed to ``fit()``, never on data
            passed to ``predict``/``score``/etc.
        class_weight: Class balancing strategy forwarded to the
            classification estimator. Ignored for regression tasks.
        classification_algorithm: Key into the classification
            estimator registry. Currently only ``"logistic_regression"``
            is available.
        regression_algorithm: Key into the regression estimator
            registry: ``"ridge"`` or ``"linear_svr"``.
        seed: Random seed forwarded to the underlying estimator.

    Raises:
        ValueError: If ``classification_algorithm`` or
            ``regression_algorithm`` is not a recognized key.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        regularization: float = 1.0,
        standardize: bool = True,
        class_weight: str | None = "balanced",
        classification_algorithm: str = "logistic_regression",
        regression_algorithm: str = "ridge",
        seed: int = 42,
    ) -> None:
        self.task_spec = task_spec
        self.regularization = regularization
        self.standardize = standardize
        self.class_weight = class_weight
        self.classification_algorithm = classification_algorithm
        self.regression_algorithm = regression_algorithm
        self.seed = seed

        self._is_classification = task_spec.label_type == "classification"
        self._n_classes: int | None = None

        self.pipeline: Pipeline = self._build_pipeline()

    def _build_estimator(self) -> Any:
        """Instantiate the estimator selected for this task's label type.

        Returns:
            An unfitted sklearn estimator instance.

        Raises:
            ValueError: If the configured algorithm name is unknown.
        """
        if self._is_classification:
            if self.classification_algorithm not in _CLASSIFICATION_ESTIMATORS:
                available = ", ".join(sorted(_CLASSIFICATION_ESTIMATORS))
                raise ValueError(
                    f"Unknown classification algorithm "
                    f"{self.classification_algorithm!r}. Available: {available}"
                )
            estimator_cls = _CLASSIFICATION_ESTIMATORS[self.classification_algorithm]
            return estimator_cls(
                C=self.regularization,
                class_weight=self.class_weight,
                max_iter=1000,
                random_state=self.seed,
            )

        if self.regression_algorithm not in _REGRESSION_ESTIMATORS:
            available = ", ".join(sorted(_REGRESSION_ESTIMATORS))
            raise ValueError(
                f"Unknown regression algorithm "
                f"{self.regression_algorithm!r}. Available: {available}"
            )
        estimator_cls = _REGRESSION_ESTIMATORS[self.regression_algorithm]
        if self.regression_algorithm == "ridge":
            return estimator_cls(alpha=self.regularization, random_state=self.seed)
        return estimator_cls(
            C=self.regularization, random_state=self.seed, max_iter=10000
        )

    def _build_pipeline(self) -> Pipeline:
        """Construct the (optionally standardizing) estimator pipeline.

        Returns:
            An unfitted ``Pipeline``. When ``standardize=True``, a
            ``StandardScaler`` step precedes the estimator so that
            ``pipeline.fit(X, y)`` fits the scaler on ``X`` alone,
            and later calls transform new data using those same
            train-fit statistics.
        """
        steps: list[tuple[str, Any]] = []
        if self.standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("estimator", self._build_estimator()))
        return Pipeline(steps)

    def fit(self, X: Any, y: Any) -> "LinearProbe":
        """Fit the probe on a single training fold.

        Args:
            X: Feature matrix of shape ``(n_samples, hidden)``. May be
                a ``torch.Tensor`` (including float16) or array-like.
            y: Target labels or values of shape ``(n_samples,)``.

        Returns:
            ``self``, to allow chaining.
        """
        X = _to_numpy(X)
        y = np.asarray(y)

        if self._is_classification:
            self._n_classes = len(np.unique(y))

        self.pipeline.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels or values for new examples.

        Args:
            X: Feature matrix of shape ``(n_samples, hidden)``.

        Returns:
            Predicted labels (classification) or values (regression).
        """
        return self.pipeline.predict(_to_numpy(X))

    def decision_function(self, X: Any) -> np.ndarray:
        """Signed distance to the decision boundary for each example.

        Only defined for classification probes.

        Args:
            X: Feature matrix of shape ``(n_samples, hidden)``.

        Returns:
            Decision function values from the underlying classifier.

        Raises:
            AttributeError: If this probe is a regression probe.
        """
        if not self._is_classification:
            raise AttributeError(
                "decision_function is only defined for classification probes"
            )
        return self.pipeline.decision_function(_to_numpy(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predicted class probabilities for each example.

        Only defined for classification probes.

        Args:
            X: Feature matrix of shape ``(n_samples, hidden)``.

        Returns:
            Array of shape ``(n_samples, n_classes)`` of predicted
            probabilities.

        Raises:
            AttributeError: If this probe is a regression probe.
        """
        if not self._is_classification:
            raise AttributeError(
                "predict_proba is only defined for classification probes"
            )
        return self.pipeline.predict_proba(_to_numpy(X))

    def score(self, X: Any, y: Any, metric: str = "accuracy") -> float:
        """Score the probe on held-out examples.

        Args:
            X: Feature matrix of shape ``(n_samples, hidden)``.
            y: True labels or values of shape ``(n_samples,)``.
            metric: For classification, ``"accuracy"`` or ``"f1"``.
                Ignored for regression, which always reports R².

        Returns:
            The requested score as a float.

        Raises:
            ValueError: If ``metric`` is unrecognized for a
                classification probe.
        """
        X = _to_numpy(X)
        y = np.asarray(y)
        predictions = self.predict(X)

        if not self._is_classification:
            return float(r2_score(y, predictions))

        if metric == "accuracy":
            return float(accuracy_score(y, predictions))
        if metric == "f1":
            n_classes = self._n_classes if self._n_classes is not None else 2
            average = "binary" if n_classes <= 2 else "macro"
            return float(f1_score(y, predictions, average=average))
        raise ValueError(
            f"Unknown classification metric {metric!r}. Expected 'accuracy' or 'f1'."
        )
