"""
Tests for control-probe selectivity computation.

Validates that the control probe is trained on labels produced by
``generate_shuffled_labels``, that the selectivity formula is exactly
``task_score - control_score``, that selectivity is near zero on a
label-independent representation, and that it is clearly positive on
well-separated, label-dependent data.
"""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.data.controls import generate_shuffled_labels
from src.data.tasks import TaskSpec
from src.probing.cross_validation import CVResult
from src.probing.probes import LinearProbe
from src.probing.selectivity import SelectivityResult, compute_selectivity


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
def probing_config() -> dict[str, Any]:
    """A minimal in-memory probing config with only the sections
    ``compute_selectivity`` (via ``run_cross_validated_probe``) consumes."""
    return {
        "probe": {
            "classification": "logistic_regression",
            "regression": "ridge",
            "standardize": True,
            "class_weight": "balanced",
            "regularization_grid": [0.001, 0.01, 0.1, 1.0, 10.0],
        },
        "cv": {"folds": 5, "seed": 42},
    }


def _make_cv_result(mean_score: float, task_spec: TaskSpec) -> CVResult:
    """Build a minimal, cheaply-constructed CVResult for formula tests."""
    probe = LinearProbe(task_spec)
    return CVResult(
        best_regularization=1.0,
        fold_scores=[mean_score] * 5,
        mean_score=mean_score,
        std_score=0.0,
        probe=probe,
    )


# ---------------------------------------------------------------------------
# TestControlUsesShuffledLabels
# ---------------------------------------------------------------------------


class TestControlUsesShuffledLabels:
    """Tests that the control run's labels come from generate_shuffled_labels."""

    def test_generate_shuffled_labels_called_with_seed_and_column(
        self,
        binary_classification_task: TaskSpec,
        probing_config: dict[str, Any],
    ) -> None:
        """generate_shuffled_labels should be called once with the
        configured cv seed and the 'label' column."""
        X, y = make_classification(
            n_samples=100, n_features=10, class_sep=3.0, random_state=0
        )

        with patch(
            "src.probing.selectivity.generate_shuffled_labels",
            wraps=generate_shuffled_labels,
        ) as spy:
            compute_selectivity(X, y, binary_classification_task, probing_config)

        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs["label_column"] == "label"
        assert kwargs["seed"] == probing_config["cv"]["seed"]

    def test_control_labels_match_direct_shuffle_call(
        self,
        binary_classification_task: TaskSpec,
        probing_config: dict[str, Any],
    ) -> None:
        """The control probe's labels should be exactly what
        generate_shuffled_labels produces for the same y and seed."""
        from datasets import Dataset

        from src.probing.cross_validation import run_cross_validated_probe

        X, y = make_classification(
            n_samples=100, n_features=10, class_sep=3.0, random_state=0
        )
        seed = probing_config["cv"]["seed"]

        expected_dataset = generate_shuffled_labels(
            Dataset.from_dict({"label": list(y), "example_id": list(range(len(y)))}),
            label_column="label",
            seed=seed,
        )
        expected_labels = np.asarray(expected_dataset["label"])

        with patch(
            "src.probing.selectivity.run_cross_validated_probe",
            wraps=run_cross_validated_probe,
        ) as spy:
            compute_selectivity(X, y, binary_classification_task, probing_config)

        # First call is the task probe (real labels), second is the
        # control probe (shuffled labels).
        _, control_call = spy.call_args_list
        control_y = control_call.args[1]
        np.testing.assert_array_equal(np.asarray(control_y), expected_labels)


# ---------------------------------------------------------------------------
# TestSelectivityFormula
# ---------------------------------------------------------------------------


class TestSelectivityFormula:
    """Tests for the selectivity = task_score - control_score formula."""

    def test_formula_matches_canned_results(
        self,
        binary_classification_task: TaskSpec,
        probing_config: dict[str, Any],
    ) -> None:
        """selectivity should equal task_score - control_score exactly,
        using canned CVResults independent of real learning dynamics."""
        X, y = make_classification(
            n_samples=50, n_features=5, class_sep=3.0, random_state=0
        )
        task_result = _make_cv_result(0.90, binary_classification_task)
        control_result = _make_cv_result(0.55, binary_classification_task)

        with patch(
            "src.probing.selectivity.run_cross_validated_probe",
            side_effect=[task_result, control_result],
        ):
            result = compute_selectivity(
                X, y, binary_classification_task, probing_config
            )

        assert result.selectivity == pytest.approx(0.35)
        assert result.task_score == pytest.approx(0.90)
        assert result.control_score == pytest.approx(0.55)

    def test_result_type(
        self,
        binary_classification_task: TaskSpec,
        probing_config: dict[str, Any],
    ) -> None:
        """compute_selectivity should return a SelectivityResult."""
        X, y = make_classification(
            n_samples=50, n_features=5, class_sep=3.0, random_state=0
        )
        result = compute_selectivity(X, y, binary_classification_task, probing_config)
        assert isinstance(result, SelectivityResult)

    def test_task_and_control_results_are_cv_results(
        self,
        binary_classification_task: TaskSpec,
        probing_config: dict[str, Any],
    ) -> None:
        """Both nested results should be full CVResult instances, so
        downstream consumers (e.g. margin/calibration) can reach the
        fitted probe without recomputation."""
        X, y = make_classification(
            n_samples=50, n_features=5, class_sep=3.0, random_state=0
        )
        result = compute_selectivity(X, y, binary_classification_task, probing_config)
        assert isinstance(result.task_result, CVResult)
        assert isinstance(result.control_result, CVResult)


# ---------------------------------------------------------------------------
# TestSelectivityOnSyntheticData
# ---------------------------------------------------------------------------


class TestSelectivityOnSyntheticData:
    """End-to-end selectivity behavior on synthetic representations."""

    def test_selectivity_near_zero_on_label_independent_representation(
        self,
        binary_classification_task: TaskSpec,
        probing_config: dict[str, Any],
    ) -> None:
        """Pure-noise features with random labels should give both the
        task and control probes chance-level performance, so their gap
        (selectivity) should be near zero."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 20))
        y = rng.integers(0, 2, size=200)

        result = compute_selectivity(X, y, binary_classification_task, probing_config)

        assert abs(result.selectivity) < 0.15

    def test_selectivity_positive_on_separable_data(
        self,
        binary_classification_task: TaskSpec,
        probing_config: dict[str, Any],
    ) -> None:
        """Well-separated, label-dependent features should let the task
        probe substantially outperform its shuffled-label control."""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=15,
            n_redundant=0,
            n_classes=2,
            class_sep=3.0,
            random_state=42,
        )

        result = compute_selectivity(X, y, binary_classification_task, probing_config)

        assert result.selectivity > 0.3


# ---------------------------------------------------------------------------
# TestRealConfig
# ---------------------------------------------------------------------------


class TestRealConfig:
    """Integration smoke test against the actual configs/probing.yaml."""

    def test_real_config_end_to_end(
        self, binary_classification_task: TaskSpec
    ) -> None:
        """The shipped probing.yaml should drive a full selectivity run
        without errors, on small synthetic data."""
        from src.probing.probes import load_probing_config

        config = load_probing_config()
        X, y = make_classification(
            n_samples=100, n_features=10, class_sep=3.0, random_state=0
        )
        result = compute_selectivity(X, y, binary_classification_task, config)
        assert -1.0 <= result.selectivity <= 1.0
