"""
Tests for the per-layer probing pipeline orchestrator.

Builds cache directories the same way the extraction pipeline does
(via ``save_representations``/``safetensors``) rather than mocking the
cache module, so pipeline tests exercise the real cache format.
Validates internal grouping/selection helpers, the tidy result
table's schema and row count, per-layer aggregation correctness, CSV
output, and prediction-depth aggregation.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.extraction.cache import save_representations
from src.probing.pipeline import (
    ProbingPipeline,
    RESULT_COLUMNS,
    _group_by_pool_strategy,
    _select_rows,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
HIDDEN_SIZE = 8
NUM_EXAMPLES = 60


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def probing_config(tmp_path: Path) -> dict[str, Any]:
    """A minimal in-memory probing config with output routed to tmp_path."""
    return {
        "probe": {
            "classification": "logistic_regression",
            "regression": "ridge",
            "standardize": True,
            "class_weight": "balanced",
            "regularization_grid": [0.01, 0.1, 1.0],
        },
        "cv": {"folds": 5, "seed": 42},
        "output_dir": str(tmp_path / "probing_output"),
    }


def _make_classification_cache(
    tmp_path: Path,
    num_layers: int = NUM_LAYERS,
    n: int = NUM_EXAMPLES,
    hidden: int = HIDDEN_SIZE,
    pool_strategy: str = "last_token",
    dataset_name: str = "sst2",
) -> Path:
    """Build a real cache directory with layer-separable binary-label data.

    Separability between the two classes increases with layer depth
    (layer 0 is pure noise, later layers are increasingly separable),
    mirroring the qualitative pattern probing is expected to detect:
    rising probe accuracy with depth.
    """
    rng = np.random.default_rng(0)
    y = [0] * (n // 2) + [1] * (n // 2)
    y_arr = np.array(y)

    representations: dict[int, torch.Tensor] = {}
    for layer in range(num_layers):
        separation = layer * 1.5
        X = rng.normal(size=(n, hidden)).astype(np.float32)
        X[:, 0] += separation * y_arr
        representations[layer] = torch.tensor(X)

    return save_representations(
        representations=representations,
        labels=y,
        example_ids=list(range(n)),
        token_counts=[10] * n,
        output_dir=str(tmp_path / "cache"),
        model_key="mock-model",
        dataset_name=dataset_name,
        split="validation",
        pool_strategy=pool_strategy,
        seed=42,
    )


def _make_regression_cache(
    tmp_path: Path,
    num_layers: int = 2,
    n: int = 40,
    hidden: int = HIDDEN_SIZE,
    pool_strategy: str = "last_token",
    dataset_name: str = "periodic_table",
) -> Path:
    """Build a real cache directory with continuous-valued labels."""
    rng = np.random.default_rng(1)
    y = list(rng.normal(size=n))

    representations: dict[int, torch.Tensor] = {
        layer: torch.tensor(rng.normal(size=(n, hidden)).astype(np.float32))
        for layer in range(num_layers)
    }

    return save_representations(
        representations=representations,
        labels=y,
        example_ids=list(range(n)),
        token_counts=[10] * n,
        output_dir=str(tmp_path / "cache_reg"),
        model_key="mock-model",
        dataset_name=dataset_name,
        split="validation",
        pool_strategy=pool_strategy,
        seed=42,
    )


@pytest.fixture
def classification_cache_dir(tmp_path: Path) -> Path:
    return _make_classification_cache(tmp_path)


@pytest.fixture
def regression_cache_dir(tmp_path: Path) -> Path:
    return _make_regression_cache(tmp_path)


# ---------------------------------------------------------------------------
# TestGroupByPoolStrategy
# ---------------------------------------------------------------------------


class TestGroupByPoolStrategy:
    """Tests for the manifest-grouping helper."""

    def test_single_strategy_groups_all_indices(self) -> None:
        """A manifest with one pool_strategy value should produce a
        single group containing every row index in order."""
        manifest = [{"pool_strategy": "last_token"} for _ in range(5)]
        groups = _group_by_pool_strategy(manifest)
        assert groups == {"last_token": [0, 1, 2, 3, 4]}

    def test_interleaved_strategies_split_correctly(self) -> None:
        """Rows should be grouped by pool_strategy regardless of how
        they're interleaved in the manifest, preserving relative order
        within each group."""
        manifest = [
            {"pool_strategy": "last_token"},
            {"pool_strategy": "mean"},
            {"pool_strategy": "last_token"},
            {"pool_strategy": "mean"},
            {"pool_strategy": "last_token"},
        ]
        groups = _group_by_pool_strategy(manifest)
        assert groups == {"last_token": [0, 2, 4], "mean": [1, 3]}

    def test_empty_manifest_gives_no_groups(self) -> None:
        """An empty manifest should produce an empty grouping."""
        assert _group_by_pool_strategy([]) == {}


# ---------------------------------------------------------------------------
# TestSelectRows
# ---------------------------------------------------------------------------


class TestSelectRows:
    """Tests for the row-selection/upcasting helper."""

    def test_selects_requested_rows_in_order(self) -> None:
        """The returned array should contain exactly the requested rows,
        in the requested order."""
        tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        selected = _select_rows(tensor, [3, 0, 4])
        expected = tensor[[3, 0, 4]].numpy()
        np.testing.assert_array_equal(selected, expected)

    def test_upcasts_fp16_to_float32(self) -> None:
        """Cached representations are fp16; selection should upcast to
        float32 for sklearn compatibility."""
        tensor = torch.randn(5, 4).half()
        selected = _select_rows(tensor, [0, 1])
        assert selected.dtype == np.float32


# ---------------------------------------------------------------------------
# TestProbingPipelineSchema
# ---------------------------------------------------------------------------


class TestProbingPipelineSchema:
    """Tests for the tidy result table's shape and column schema."""

    def test_row_count_matches_layers_times_pool_strategies(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """One cached pool strategy x NUM_LAYERS layers should produce
        exactly NUM_LAYERS rows."""
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(classification_cache_dir)
        assert len(rows) == NUM_LAYERS

    def test_row_keys_match_result_columns(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """Every row should have exactly the documented column set."""
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(classification_cache_dir)
        for row in rows:
            assert set(row.keys()) == set(RESULT_COLUMNS)

    def test_identifying_columns_are_correct(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """model/task/pool_strategy should be constant across rows, and
        layer should cover every cached layer index exactly once."""
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(classification_cache_dir)

        assert {row["model"] for row in rows} == {"mock-model"}
        assert {row["task"] for row in rows} == {"sst2"}
        assert {row["pool_strategy"] for row in rows} == {"last_token"}
        assert sorted(row["layer"] for row in rows) == list(range(NUM_LAYERS))

    def test_raises_on_empty_cache_dir(
        self, tmp_path: Path, probing_config: dict[str, Any]
    ) -> None:
        """A cache directory with no safetensors/manifest should raise
        a clear error rather than silently producing an empty table."""
        empty_dir = tmp_path / "empty_cache"
        empty_dir.mkdir()
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        with pytest.raises(ValueError, match="No cached representations"):
            pipeline.run(empty_dir)


# ---------------------------------------------------------------------------
# TestProbingPipelineAggregation
# ---------------------------------------------------------------------------


class TestProbingPipelineAggregation:
    """Tests for per-layer metric correctness and classification/regression
    dispatch."""

    def test_score_rises_with_layer_separability(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """The deepest (most separable) layer should score well above
        the shallowest (pure-noise) layer."""
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(classification_cache_dir)
        by_layer = {row["layer"]: row for row in rows}

        assert by_layer[NUM_LAYERS - 1]["score"] > by_layer[0]["score"]

    def test_classification_margin_and_ece_are_populated(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """margin_mean and ece should be finite, non-NaN numbers for a
        classification task."""
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(classification_cache_dir)
        for row in rows:
            assert np.isfinite(row["margin_mean"])
            assert np.isfinite(row["ece"])

    def test_regression_margin_and_ece_are_nan(
        self, regression_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """margin_mean and ece are undefined for regression probes and
        should be reported as NaN rather than raising."""
        pipeline = ProbingPipeline("mock-model", "periodic_table", probing_config)
        rows = pipeline.run(regression_cache_dir)
        for row in rows:
            assert np.isnan(row["margin_mean"])
            assert np.isnan(row["ece"])

    def test_regression_score_is_r_squared_range(
        self, regression_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """Regression score should be a valid R^2 value (no lower bound,
        upper-bounded by 1)."""
        pipeline = ProbingPipeline("mock-model", "periodic_table", probing_config)
        rows = pipeline.run(regression_cache_dir)
        for row in rows:
            assert row["score"] <= 1.0
