"""
Tests for probing metrics: classification margin, expected calibration
error, and prediction depth.

Validates margin sign and formula correctness for both binary and
multi-class probes, expected calibration error on constructed
perfectly-calibrated and miscalibrated confidence distributions, and
prediction-depth correctness on constructed stable and unstable
per-layer prediction sequences.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from src.data.tasks import TaskSpec
from src.metrics.calibration import expected_calibration_error
from src.metrics.margins import classification_margin
from src.metrics.prediction_depth import prediction_depth
from src.probing.probes import LinearProbe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_classification_task() -> TaskSpec:
    """A mock binary classification task, independent of tasks.yaml."""
    return TaskSpec(
        name="mock_binary_cls",
        family="shallow_semantic",
        label_type="classification",
        num_classes=2,
        extraction_position="last_token",
        description="mock binary classification task",
    )


@pytest.fixture
def multiclass_classification_task() -> TaskSpec:
    """A mock 3-class classification task."""
    return TaskSpec(
        name="mock_multiclass_cls",
        family="factual_lookup",
        label_type="classification",
        num_classes=3,
        extraction_position="last_token",
        description="mock multiclass classification task",
    )


@pytest.fixture
def regression_task() -> TaskSpec:
    """A mock regression task."""
    return TaskSpec(
        name="mock_regression",
        family="factual_lookup",
        label_type="regression",
        num_classes=None,
        extraction_position="last_token",
        description="mock regression task",
    )


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Well-separated binary classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=0,
        n_classes=2,
        class_sep=3.0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def overlapping_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Noisy, overlapping binary classification data with real errors."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        class_sep=0.5,
        flip_y=0.25,
        random_state=7,
    )
    return X, y


@pytest.fixture
def multiclass_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Well-separated 3-class classification data."""
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=15,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=3.0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Low-noise linear regression data."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=8,
        noise=1.0,
        random_state=42,
    )
    return X, y


def _mock_probe(task_spec: TaskSpec, probabilities: np.ndarray, predictions: np.ndarray) -> LinearProbe:
    """Build a LinearProbe whose predict/predict_proba return canned arrays.

    Lets ECE be tested against exact, hand-picked confidence/accuracy
    combinations, independent of real learning dynamics.
    """
    probe = LinearProbe(task_spec)
    probe.predict_proba = MagicMock(return_value=probabilities)
    probe.predict = MagicMock(return_value=predictions)
    return probe


# ---------------------------------------------------------------------------
# TestClassificationMargin
# ---------------------------------------------------------------------------


class TestClassificationMargin:
    """Tests for classification_margin on binary and multi-class probes."""

    def test_binary_margin_matches_signed_decision_function(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Binary margin should equal decision_function signed by whether
        the true label is the positive (classes_[1]) class."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)

        margins = classification_margin(probe, X, y)
        raw = probe.decision_function(X)
        classes = probe.pipeline.named_steps["estimator"].classes_
        expected_sign = np.where(y == classes[1], 1.0, -1.0)

        np.testing.assert_allclose(margins, expected_sign * raw)

    def test_binary_margin_shape(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Margin should return one value per example."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        margins = classification_margin(probe, X, y)
        assert margins.shape == (X.shape[0],)

    def test_binary_margin_positive_when_correctly_classified(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Correctly classified examples should have a positive margin."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        predictions = probe.predict(X)
        margins = classification_margin(probe, X, y)

        correct = predictions == y
        assert np.all(margins[correct] > 0)

    def test_binary_margin_negative_when_misclassified(
        self,
        binary_classification_task: TaskSpec,
        overlapping_classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Misclassified examples should have a negative margin, on data
        noisy enough to guarantee real classifier errors."""
        X, y = overlapping_classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        predictions = probe.predict(X)
        margins = classification_margin(probe, X, y)

        incorrect = predictions != y
        assert incorrect.sum() > 0
        assert np.all(margins[incorrect] < 0)

    def test_binary_margin_larger_for_more_confident_examples(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Examples further from the boundary should have a larger
        magnitude margin than examples closer to it."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        margins = classification_margin(probe, X, y)
        raw = probe.decision_function(X)

        far_idx = np.argmax(np.abs(raw))
        near_idx = np.argmin(np.abs(raw))
        assert abs(margins[far_idx]) > abs(margins[near_idx])

    def test_multiclass_margin_matches_true_minus_max_other_formula(
        self,
        multiclass_classification_task: TaskSpec,
        multiclass_classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Multi-class margin should equal the true class's decision score
        minus the highest score among the other classes, recomputed
        directly from the raw decision_function matrix."""
        X, y = multiclass_classification_data
        probe = LinearProbe(multiclass_classification_task).fit(X, y)

        margins = classification_margin(probe, X, y)
        scores = probe.decision_function(X)
        classes = probe.pipeline.named_steps["estimator"].classes_
        class_to_idx = {label: idx for idx, label in enumerate(classes)}

        expected = np.empty(len(y))
        for i, label in enumerate(y):
            true_idx = class_to_idx[label]
            true_score = scores[i, true_idx]
            other_scores = np.delete(scores[i], true_idx)
            expected[i] = true_score - other_scores.max()

        np.testing.assert_allclose(margins, expected)

    def test_multiclass_margin_positive_when_correctly_classified(
        self,
        multiclass_classification_task: TaskSpec,
        multiclass_classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Correctly classified multi-class examples should have a
        positive margin (the true class wins its one-vs-rest vote)."""
        X, y = multiclass_classification_data
        probe = LinearProbe(multiclass_classification_task).fit(X, y)
        predictions = probe.predict(X)
        margins = classification_margin(probe, X, y)

        correct = predictions == y
        assert np.all(margins[correct] > 0)

    def test_margin_raises_for_regression_probe(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Margin should be undefined for regression probes."""
        X, y = regression_data
        probe = LinearProbe(regression_task).fit(X, y)
        with pytest.raises(AttributeError):
            classification_margin(probe, X, y)


# ---------------------------------------------------------------------------
# TestExpectedCalibrationError
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error on constructed confidence data."""

    def test_perfectly_calibrated_single_bin_gives_zero_ece(
        self, binary_classification_task: TaskSpec
    ) -> None:
        """When a bin's mean confidence exactly equals its accuracy, ECE
        should be zero."""
        n = 20
        confidence = 0.8
        probabilities = np.tile([1 - confidence, confidence], (n, 1))
        y = np.array([0, 1] * (n // 2))
        predictions = y.copy()
        predictions[:4] = 1 - predictions[:4]  # 4 of 20 wrong -> accuracy 0.8

        probe = _mock_probe(binary_classification_task, probabilities, predictions)
        ece = expected_calibration_error(probe, np.zeros((n, 1)), y, n_bins=10)

        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_miscalibrated_single_bin_gives_expected_gap(
        self, binary_classification_task: TaskSpec
    ) -> None:
        """A bin with high confidence but low accuracy should report the
        exact |accuracy - confidence| gap."""
        n = 20
        confidence = 0.95
        probabilities = np.tile([1 - confidence, confidence], (n, 1))
        y = np.array([0, 1] * (n // 2))
        predictions = y.copy()
        predictions[:10] = 1 - predictions[:10]  # 10 of 20 wrong -> accuracy 0.5

        probe = _mock_probe(binary_classification_task, probabilities, predictions)
        ece = expected_calibration_error(probe, np.zeros((n, 1)), y, n_bins=10)

        assert ece == pytest.approx(0.45, abs=1e-9)

    def test_weighted_average_across_multiple_bins(
        self, binary_classification_task: TaskSpec
    ) -> None:
        """ECE across two populated bins should equal the sample-count
        weighted average of each bin's gap, not a plain mean of bins."""
        # Bin A: 15 examples, confidence 0.6, all correct (gap = 0.4).
        conf_a = np.full(15, 0.6)
        pred_a = np.ones(15)
        y_a = np.ones(15)

        # Bin B: 5 examples, confidence 0.95, all wrong (gap = 0.95).
        conf_b = np.full(5, 0.95)
        pred_b = np.ones(5)
        y_b = np.zeros(5)

        confidences = np.concatenate([conf_a, conf_b])
        probabilities = np.column_stack([1 - confidences, confidences])
        predictions = np.concatenate([pred_a, pred_b])
        y = np.concatenate([y_a, y_b])

        probe = _mock_probe(binary_classification_task, probabilities, predictions)
        ece = expected_calibration_error(probe, np.zeros((20, 1)), y, n_bins=10)

        expected = (15 / 20) * 0.4 + (5 / 20) * 0.95
        assert ece == pytest.approx(expected, abs=1e-9)

    def test_well_calibrated_on_real_separable_data(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """A probe fit on well-separated, high-accuracy data should be
        reasonably well calibrated end to end."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        ece = expected_calibration_error(probe, X, y)
        assert 0.0 <= ece < 0.2

    def test_raises_for_regression_probe(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """ECE should be undefined for regression probes."""
        X, y = regression_data
        probe = LinearProbe(regression_task).fit(X, y)
        with pytest.raises(ValueError, match="only defined for classification"):
            expected_calibration_error(probe, X, y)

    def test_custom_n_bins_is_respected(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """A different n_bins value should run without error and stay
        within the valid [0, 1] range."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        ece = expected_calibration_error(probe, X, y, n_bins=5)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# TestPredictionDepth
# ---------------------------------------------------------------------------


class TestPredictionDepth:
    """Tests for prediction_depth on constructed per-layer prediction
    sequences."""

    def test_all_layers_agree_returns_zero(self) -> None:
        """A sequence that never changes should stabilize at layer 0."""
        assert prediction_depth([1, 1, 1, 1]) == 0

    def test_stabilizes_partway_through(self) -> None:
        """The depth should be the first layer of the run that matches
        the final layer's prediction, ignoring earlier layers."""
        assert prediction_depth([0, 0, 1, 1, 1]) == 2

    def test_only_final_layer_matches_itself(self) -> None:
        """When every layer but the last disagrees with the final
        prediction, depth should fall back to the deepest layer."""
        assert prediction_depth([0, 0, 0, 1]) == 3

    def test_single_layer_returns_zero(self) -> None:
        """A single-layer sequence trivially stabilizes at layer 0."""
        assert prediction_depth([7]) == 0

    def test_ignores_matches_broken_by_an_intervening_mismatch(self) -> None:
        """An early layer that happens to match the final prediction
        should not count if a later layer breaks the run — only the
        contiguous suffix matters."""
        assert prediction_depth([1, 0, 1, 1, 1]) == 2

    def test_empty_sequence_raises(self) -> None:
        """An empty prediction sequence has no layers to scan."""
        with pytest.raises(ValueError):
            prediction_depth([])

    def test_explicit_final_prediction_overrides_default(self) -> None:
        """An explicit final_prediction that differs from the sequence's
        last element should be used as the comparison target."""
        assert prediction_depth([5, 5, 3, 3], final_prediction=5) == 3

    def test_explicit_final_prediction_matching_default_behaves_the_same(self) -> None:
        """Passing the sequence's own last element as final_prediction
        should behave identically to the default."""
        assert prediction_depth([0, 0, 1, 1, 1], final_prediction=1) == 2

    def test_works_with_non_numeric_labels(self) -> None:
        """Predictions may be arbitrary class labels, not just integers."""
        assert prediction_depth(["cat", "cat", "dog", "dog"]) == 2

    def test_returns_python_int(self) -> None:
        """The returned depth should be a plain int, regardless of the
        prediction sequence's element type."""
        depth = prediction_depth(np.array([0, 0, 1, 1]))
        assert isinstance(depth, int)
