"""
Tests for the linear probe wrapper (classification + regression).

Validates estimator dispatch by task label_type, the low-capacity
linear-only guarantee, standardization scoped to the training fold
only, score-metric correctness, decision_function/predict_proba
pass-throughs, torch tensor input handling, and probing config
loading.
"""

import inspect
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

from src.data.tasks import TaskSpec
from src.probing.probes import (
    _CLASSIFICATION_ESTIMATORS,
    _REGRESSION_ESTIMATORS,
    LinearProbe,
    load_probing_config,
)


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
        n_samples=200,
        n_features=20,
        n_informative=15,
        noise=1.0,
        random_state=42,
    )
    return X, y


# ---------------------------------------------------------------------------
# TestEstimatorDispatch
# ---------------------------------------------------------------------------


class TestEstimatorDispatch:
    """Tests for label_type-keyed estimator dispatch."""

    def test_classification_task_uses_logistic_regression(
        self, binary_classification_task: TaskSpec
    ) -> None:
        """Classification tasks should dispatch to LogisticRegression."""
        probe = LinearProbe(binary_classification_task)
        assert isinstance(probe.pipeline.named_steps["estimator"], LogisticRegression)

    def test_regression_task_uses_ridge_by_default(
        self, regression_task: TaskSpec
    ) -> None:
        """Regression tasks should dispatch to Ridge by default."""
        probe = LinearProbe(regression_task)
        assert isinstance(probe.pipeline.named_steps["estimator"], Ridge)

    def test_regression_task_can_use_linear_svr(
        self, regression_task: TaskSpec
    ) -> None:
        """Regression tasks should be able to opt into LinearSVR."""
        probe = LinearProbe(regression_task, regression_algorithm="linear_svr")
        assert isinstance(probe.pipeline.named_steps["estimator"], LinearSVR)

    def test_unknown_classification_algorithm_raises(
        self, binary_classification_task: TaskSpec
    ) -> None:
        """An unrecognized classification algorithm should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown classification algorithm"):
            LinearProbe(binary_classification_task, classification_algorithm="mlp")

    def test_unknown_regression_algorithm_raises(
        self, regression_task: TaskSpec
    ) -> None:
        """An unrecognized regression algorithm should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown regression algorithm"):
            LinearProbe(regression_task, regression_algorithm="kernel_svr")


# ---------------------------------------------------------------------------
# TestLowCapacityGuarantee
# ---------------------------------------------------------------------------


class TestLowCapacityGuarantee:
    """Tests enforcing the strictly-linear, low-capacity probe constraint."""

    def test_classification_registry_is_linear_only(self) -> None:
        """The classification registry should contain only linear models."""
        assert set(_CLASSIFICATION_ESTIMATORS.values()) == {LogisticRegression}

    def test_regression_registry_is_linear_only(self) -> None:
        """The regression registry should contain only linear models."""
        assert set(_REGRESSION_ESTIMATORS.values()) == {Ridge, LinearSVR}

    def test_no_kernel_or_hidden_layer_params_in_constructor(self) -> None:
        """LinearProbe should expose no kernel or hidden-layer options."""
        params = inspect.signature(LinearProbe.__init__).parameters
        for name in params:
            assert "kernel" not in name.lower()
            assert "hidden" not in name.lower()


# ---------------------------------------------------------------------------
# TestStandardization
# ---------------------------------------------------------------------------


class TestStandardization:
    """Tests for standardization and its scoping to the training fold."""

    def test_standardize_true_adds_scaler_step(
        self, binary_classification_task: TaskSpec
    ) -> None:
        """standardize=True should insert a scaler pipeline step."""
        probe = LinearProbe(binary_classification_task, standardize=True)
        assert "scaler" in probe.pipeline.named_steps

    def test_standardize_false_omits_scaler_step(
        self, binary_classification_task: TaskSpec
    ) -> None:
        """standardize=False should omit the scaler pipeline step."""
        probe = LinearProbe(binary_classification_task, standardize=False)
        assert "scaler" not in probe.pipeline.named_steps

    def test_scaler_fit_only_on_training_fold(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """The scaler's statistics should reflect only the training fold."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )

        probe = LinearProbe(binary_classification_task)
        probe.fit(X_train, y_train)

        scaler = probe.pipeline.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-6)
        # Combined train+test statistics should differ from train-only
        # statistics (otherwise this test can't distinguish leakage).
        combined_mean = np.concatenate([X_train, X_test]).mean(axis=0)
        assert not np.allclose(scaler.mean_, combined_mean)

    def test_predict_does_not_refit_scaler(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Calling predict on new data should not change scaler statistics."""
        X, y = classification_data
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=0
        )

        probe = LinearProbe(binary_classification_task)
        probe.fit(X_train, y_train)
        scaler = probe.pipeline.named_steps["scaler"]
        mean_before = scaler.mean_.copy()

        probe.predict(X_test)

        np.testing.assert_array_equal(scaler.mean_, mean_before)


# ---------------------------------------------------------------------------
# TestFitPredictScore
# ---------------------------------------------------------------------------


class TestFitPredictScore:
    """Tests for fit/predict/score behavior and metric correctness."""

    def test_fit_returns_self(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """fit() should return the probe instance for chaining."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task)
        result = probe.fit(X, y)
        assert result is probe

    def test_predict_output_shape(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """predict() should return one prediction per example."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        predictions = probe.predict(X)
        assert predictions.shape == (X.shape[0],)

    def test_classification_score_default_is_accuracy(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """The default classification score should equal accuracy."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        probe = LinearProbe(binary_classification_task).fit(X_train, y_train)

        expected = accuracy_score(y_test, probe.predict(X_test))
        assert probe.score(X_test, y_test) == pytest.approx(expected)

    def test_classification_score_f1_metric_binary(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """metric='f1' should equal sklearn's binary f1_score."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        probe = LinearProbe(binary_classification_task).fit(X_train, y_train)

        expected = f1_score(y_test, probe.predict(X_test), average="binary")
        assert probe.score(X_test, y_test, metric="f1") == pytest.approx(expected)

    def test_classification_score_f1_metric_multiclass(
        self,
        multiclass_classification_task: TaskSpec,
        multiclass_classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """metric='f1' should equal sklearn's macro f1_score for >2 classes."""
        X, y = multiclass_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        probe = LinearProbe(multiclass_classification_task).fit(X_train, y_train)

        expected = f1_score(y_test, probe.predict(X_test), average="macro")
        assert probe.score(X_test, y_test, metric="f1") == pytest.approx(expected)

    def test_regression_score_is_r2(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Regression score should equal R^2, regardless of the metric arg."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        probe = LinearProbe(regression_task).fit(X_train, y_train)

        expected = r2_score(y_test, probe.predict(X_test))
        assert probe.score(X_test, y_test) == pytest.approx(expected)

    def test_regression_ignores_metric_argument(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """An unrecognized metric string should not raise for regression."""
        X, y = regression_data
        probe = LinearProbe(regression_task).fit(X, y)
        # Should not raise, and should still return R^2.
        assert probe.score(X, y, metric="not_a_real_metric") == pytest.approx(
            probe.score(X, y)
        )

    def test_unknown_classification_metric_raises(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """An unrecognized classification metric should raise ValueError."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        with pytest.raises(ValueError, match="Unknown classification metric"):
            probe.score(X, y, metric="mse")

    def test_high_accuracy_on_separable_classification_data(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """A well-separated dataset should score highly."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        probe = LinearProbe(binary_classification_task).fit(X_train, y_train)
        assert probe.score(X_test, y_test) > 0.85

    def test_high_r2_on_low_noise_regression_data(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Low-noise linear regression data should score a high R^2."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        probe = LinearProbe(regression_task).fit(X_train, y_train)
        assert probe.score(X_test, y_test) > 0.85


# ---------------------------------------------------------------------------
# TestDecisionFunctionAndProba
# ---------------------------------------------------------------------------


class TestDecisionFunctionAndProba:
    """Tests for decision_function/predict_proba pass-throughs."""

    def test_decision_function_shape_binary(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Binary decision_function should return one value per example."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        scores = probe.decision_function(X)
        assert scores.shape == (X.shape[0],)

    def test_decision_function_shape_multiclass(
        self,
        multiclass_classification_task: TaskSpec,
        multiclass_classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Multiclass decision_function should return one score per class."""
        X, y = multiclass_classification_data
        probe = LinearProbe(multiclass_classification_task).fit(X, y)
        scores = probe.decision_function(X)
        assert scores.shape == (X.shape[0], 3)

    def test_predict_proba_rows_sum_to_one(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Predicted probabilities should sum to 1 across classes."""
        X, y = classification_data
        probe = LinearProbe(binary_classification_task).fit(X, y)
        probs = probe.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(X.shape[0]), rtol=1e-6)

    def test_decision_function_raises_for_regression(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """decision_function should be undefined for regression probes."""
        X, y = regression_data
        probe = LinearProbe(regression_task).fit(X, y)
        with pytest.raises(AttributeError):
            probe.decision_function(X)

    def test_predict_proba_raises_for_regression(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """predict_proba should be undefined for regression probes."""
        X, y = regression_data
        probe = LinearProbe(regression_task).fit(X, y)
        with pytest.raises(AttributeError):
            probe.predict_proba(X)


# ---------------------------------------------------------------------------
# TestTensorInput
# ---------------------------------------------------------------------------


class TestTensorInput:
    """Tests for accepting torch tensors (matching the extraction cache)."""

    def test_accepts_torch_float16_tensor(
        self,
        binary_classification_task: TaskSpec,
        classification_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """fit/predict should work directly on float16 torch tensors."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        X_train_t = torch.tensor(X_train, dtype=torch.float16)
        X_test_t = torch.tensor(X_test, dtype=torch.float16)

        probe = LinearProbe(binary_classification_task)
        probe.fit(X_train_t, y_train)
        predictions = probe.predict(X_test_t)

        assert predictions.shape == (X_test.shape[0],)
        assert probe.score(X_test_t, y_test) > 0.8

    def test_accepts_numpy_array(
        self,
        regression_task: TaskSpec,
        regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """fit/predict should work directly on plain numpy arrays."""
        X, y = regression_data
        probe = LinearProbe(regression_task)
        probe.fit(X, y)
        predictions = probe.predict(X)
        assert isinstance(predictions, np.ndarray)


# ---------------------------------------------------------------------------
# TestLoadProbingConfig
# ---------------------------------------------------------------------------

SAMPLE_YAML = textwrap.dedent("""\
    probe:
      classification: "logistic_regression"
      regression: "ridge"
      standardize: true
      class_weight: "balanced"
      regularization_grid: [0.001, 0.01, 0.1, 1.0, 10.0]

    cv:
      folds: 5
      seed: 42

    models: ["pythia-160m"]
    tasks: ["sst2", "lama_trex"]

    output_dir: "outputs/probing"
""")


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal probing config YAML to a temp file.

    Returns:
        Path to the temporary YAML file.
    """
    config_file = tmp_path / "probing.yaml"
    config_file.write_text(SAMPLE_YAML)
    return config_file


class TestLoadProbingConfig:
    """Tests for load_probing_config()."""

    def test_loads_all_sections(self, sample_config: Path) -> None:
        """All top-level sections should be present in the loaded config."""
        config = load_probing_config(sample_config)
        assert set(config.keys()) == {
            "probe",
            "cv",
            "models",
            "tasks",
            "output_dir",
        }

    def test_probe_section_values(self, sample_config: Path) -> None:
        """The probe section should match the YAML exactly."""
        config = load_probing_config(sample_config)
        probe_cfg = config["probe"]
        assert probe_cfg["classification"] == "logistic_regression"
        assert probe_cfg["regression"] == "ridge"
        assert probe_cfg["standardize"] is True
        assert probe_cfg["class_weight"] == "balanced"
        assert probe_cfg["regularization_grid"] == [0.001, 0.01, 0.1, 1.0, 10.0]

    def test_cv_section_values(self, sample_config: Path) -> None:
        """The cv section should match the YAML exactly."""
        config = load_probing_config(sample_config)
        assert config["cv"] == {"folds": 5, "seed": 42}

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        """Loading from a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_probing_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# TestRealConfig
# ---------------------------------------------------------------------------


class TestRealConfig:
    """Tests against the actual configs/probing.yaml shipped in the repo."""

    def test_real_config_loads(self) -> None:
        """The shipped probing.yaml should load without errors."""
        config = load_probing_config()
        assert set(config.keys()) == {
            "probe",
            "cv",
            "models",
            "tasks",
            "output_dir",
        }

    def test_real_config_values(self) -> None:
        """The shipped config should match the values specified in the plan."""
        config = load_probing_config()
        assert config["probe"]["classification"] == "logistic_regression"
        assert config["probe"]["regression"] == "ridge"
        assert config["probe"]["standardize"] is True
        assert config["probe"]["class_weight"] == "balanced"
        assert config["probe"]["regularization_grid"] == [0.001, 0.01, 0.1, 1.0, 10.0]
        assert config["cv"]["folds"] == 5
        assert config["cv"]["seed"] == 42
        assert config["models"] == ["pythia-160m"]
        assert config["tasks"] == ["sst2", "lama_trex"]
        assert config["output_dir"] == "outputs/probing"
