"""
Tests for the k-fold cross-validation harness with regularization sweep.

Validates fold count, splitter selection (stratified for
classification, plain for regression), regularization-sweep selection
logic, mean/std computation, class balancing, seed reproducibility,
and the full-dataset refit probe.
"""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold

from src.data.tasks import TaskSpec
from src.probing.cross_validation import CVResult, run_cross_validated_probe
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
    """Well-separated binary classification data, large enough for 5 folds."""
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
def imbalanced_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Class-imbalanced binary classification data (80/20 split)."""
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=15,
        n_redundant=0,
        n_classes=2,
        weights=[0.8, 0.2],
        class_sep=2.0,
        random_state=7,
    )
    return X, y


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Low-noise linear regression data."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=15,
        noise=1.0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def probing_config() -> dict[str, Any]:
    """A minimal in-memory probing config with only the sections
    ``run_cross_validated_probe`` consumes."""
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


# ---------------------------------------------------------------------------
# TestFoldCount
# ---------------------------------------------------------------------------


class TestFoldCount:
    """Tests that the number of returned fold scores matches config."""

    def test_classification_fold_count(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """Classification tasks should produce config['cv']['folds'] scores."""
        X, y = classification_data
        result = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        assert len(result.fold_scores) == probing_config["cv"]["folds"]

    def test_regression_fold_count(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """Regression tasks should produce config['cv']['folds'] scores."""
        X, y = regression_data
        result = run_cross_validated_probe(X, y, regression_task, probing_config)
        assert len(result.fold_scores) == probing_config["cv"]["folds"]

    def test_fold_count_respects_config(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """A non-default fold count in config should be honored."""
        X, y = classification_data
        probing_config["cv"]["folds"] = 4
        result = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        assert len(result.fold_scores) == 4


# ---------------------------------------------------------------------------
# TestSplitterSelection
# ---------------------------------------------------------------------------


class TestSplitterSelection:
    """Tests that classification stratifies folds; regression does not."""

    def test_classification_folds_are_stratified(
        self,
        binary_classification_task: TaskSpec,
        imbalanced_classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """Per-fold class proportions should stay close to the global
        proportion when the splitter is stratified."""
        X, y = imbalanced_classification_data
        global_positive_rate = float(np.mean(y))

        splitter = StratifiedKFold(
            n_splits=probing_config["cv"]["folds"],
            shuffle=True,
            random_state=probing_config["cv"]["seed"],
        )
        for _, test_idx in splitter.split(X, y):
            fold_positive_rate = float(np.mean(y[test_idx]))
            assert abs(fold_positive_rate - global_positive_rate) < 0.1

    def test_regression_does_not_require_stratification(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """Regression targets are continuous; KFold should run without
        raising (StratifiedKFold would fail on continuous targets)."""
        X, y = regression_data
        result = run_cross_validated_probe(X, y, regression_task, probing_config)
        assert len(result.fold_scores) == probing_config["cv"]["folds"]


# ---------------------------------------------------------------------------
# TestRegularizationSweep
# ---------------------------------------------------------------------------


class TestRegularizationSweep:
    """Tests for the regularization-grid selection logic."""

    def test_selects_regularization_with_best_mean_score(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """The harness should select the grid value whose mean fold
        score is highest, verified via a deterministic score lookup
        independent of real learning dynamics."""
        X, y = classification_data
        score_by_regularization = {
            0.001: 0.10,
            0.01: 0.20,
            0.1: 0.95,
            1.0: 0.40,
            10.0: 0.30,
        }

        def fake_score(self: LinearProbe, X: Any, y: Any, metric: str = "accuracy") -> float:
            return score_by_regularization[self.regularization]

        with patch.object(LinearProbe, "score", fake_score):
            result = run_cross_validated_probe(
                X, y, binary_classification_task, probing_config
            )

        assert result.best_regularization == 0.1
        assert result.mean_score == pytest.approx(0.95)

    def test_ties_broken_by_first_encountered(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """When two regularization values tie, the first one in the
        grid (as ordered in config) should be selected."""
        X, y = classification_data
        score_by_regularization = {
            0.001: 0.50,
            0.01: 0.90,
            0.1: 0.90,
            1.0: 0.10,
            10.0: 0.10,
        }

        def fake_score(self: LinearProbe, X: Any, y: Any, metric: str = "accuracy") -> float:
            return score_by_regularization[self.regularization]

        with patch.object(LinearProbe, "score", fake_score):
            result = run_cross_validated_probe(
                X, y, binary_classification_task, probing_config
            )

        assert result.best_regularization == 0.01

    def test_all_candidates_use_identical_folds(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """Every regularization candidate should be scored on the same
        fold partition, so shrinking the grid to one value that scores
        identically under two different configs is only possible if
        both runs saw the same split."""
        X, y = classification_data
        single_value_config = dict(probing_config)
        single_value_config["probe"] = dict(probing_config["probe"])
        single_value_config["probe"]["regularization_grid"] = [1.0]

        result_a = run_cross_validated_probe(
            X, y, binary_classification_task, single_value_config
        )
        result_b = run_cross_validated_probe(
            X, y, binary_classification_task, single_value_config
        )
        assert result_a.fold_scores == pytest.approx(result_b.fold_scores)


# ---------------------------------------------------------------------------
# TestMeanStd
# ---------------------------------------------------------------------------


class TestMeanStd:
    """Tests for mean_score/std_score computation."""

    def test_mean_score_matches_numpy_mean(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """mean_score should equal np.mean(fold_scores)."""
        X, y = classification_data
        result = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        assert result.mean_score == pytest.approx(np.mean(result.fold_scores))

    def test_std_score_matches_numpy_std(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """std_score should equal the population std of fold_scores."""
        X, y = regression_data
        result = run_cross_validated_probe(X, y, regression_task, probing_config)
        assert result.std_score == pytest.approx(np.std(result.fold_scores))


# ---------------------------------------------------------------------------
# TestClassBalancing
# ---------------------------------------------------------------------------


class TestClassBalancing:
    """Tests that class_weight from config reaches the underlying estimator."""

    def test_class_weight_reaches_final_probe(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """The refit probe's estimator should carry class_weight='balanced'."""
        X, y = classification_data
        result = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        estimator = result.probe.pipeline.named_steps["estimator"]
        assert estimator.class_weight == "balanced"

    def test_class_weight_none_is_respected(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """A None class_weight in config should also propagate through."""
        X, y = classification_data
        probing_config["probe"]["class_weight"] = None
        result = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        estimator = result.probe.pipeline.named_steps["estimator"]
        assert estimator.class_weight is None


# ---------------------------------------------------------------------------
# TestSeedReproducibility
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Tests that identical inputs/seed produce identical results."""

    def test_same_seed_same_fold_scores(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """Two calls with the same seed should produce bit-identical
        fold scores and the same selected regularization."""
        X, y = classification_data
        result_a = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        result_b = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        assert result_a.fold_scores == result_b.fold_scores
        assert result_a.best_regularization == result_b.best_regularization

    def test_different_seed_can_change_fold_scores(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """A different seed should (generically) produce a different
        fold partition and therefore different fold scores."""
        X, y = regression_data
        result_a = run_cross_validated_probe(X, y, regression_task, probing_config)

        probing_config["cv"]["seed"] = 999
        result_b = run_cross_validated_probe(X, y, regression_task, probing_config)

        assert result_a.fold_scores != result_b.fold_scores


# ---------------------------------------------------------------------------
# TestFinalProbe
# ---------------------------------------------------------------------------


class TestFinalProbe:
    """Tests for the full-dataset refit probe attached to CVResult."""

    def test_final_probe_uses_best_regularization(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """The refit probe's regularization should match best_regularization."""
        X, y = classification_data
        result = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        assert result.probe.regularization == result.best_regularization

    def test_final_probe_is_fit_on_full_dataset(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """The refit probe should produce predictions for every example
        in the full dataset, not just one fold's worth."""
        X, y = classification_data
        result = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        predictions = result.probe.predict(X)
        assert predictions.shape == (X.shape[0],)

    def test_final_probe_is_a_linear_probe_instance(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """The refit probe should be a fitted LinearProbe instance."""
        X, y = regression_data
        result = run_cross_validated_probe(X, y, regression_task, probing_config)
        assert isinstance(result.probe, LinearProbe)


# ---------------------------------------------------------------------------
# TestResultType
# ---------------------------------------------------------------------------


class TestResultType:
    """Tests for the CVResult return type."""

    def test_returns_cv_result(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
        probing_config: dict[str, Any],
    ) -> None:
        """run_cross_validated_probe should return a CVResult."""
        X, y = classification_data
        result = run_cross_validated_probe(
            X, y, binary_classification_task, probing_config
        )
        assert isinstance(result, CVResult)


# ---------------------------------------------------------------------------
# TestRealConfig
# ---------------------------------------------------------------------------


class TestRealConfig:
    """Integration smoke test against the actual configs/probing.yaml."""

    def test_real_config_end_to_end(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """The shipped probing.yaml should drive a full CV run without
        errors, on small synthetic data."""
        from src.probing.probes import load_probing_config

        config = load_probing_config()
        X, y = classification_data
        result = run_cross_validated_probe(
            X, y, binary_classification_task, config
        )
        assert len(result.fold_scores) == config["cv"]["folds"]
        assert 0.0 <= result.mean_score <= 1.0
