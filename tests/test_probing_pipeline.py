"""
Tests for the per-layer probing pipeline orchestrator.

Builds cache directories the same way the extraction pipeline does
(via ``save_representations``/``safetensors``) rather than mocking the
cache module, so pipeline tests exercise the real cache format.
Validates internal grouping/selection helpers, the tidy result
table's schema and row count, per-layer aggregation correctness, CSV
output, and prediction-depth aggregation.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.extraction.cache import save_representations
from src.metrics.prediction_depth import prediction_depth
from src.probing.pipeline import (
    ProbingPipeline,
    RESULT_COLUMNS,
    _group_by_pool_strategy,
    _select_rows,
    _stack_prediction_depths,
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


def _make_multi_pool_strategy_cache(
    tmp_path: Path,
    n_per_group: int = 20,
    hidden: int = HIDDEN_SIZE,
    layer_indices: tuple[int, ...] = (0, 1),
) -> Path:
    """Build a cache directory with two pool strategies in one manifest.

    The public save_representations API always writes a single
    pool_strategy per call (and truncates the manifest on each call,
    so calling it twice cannot accumulate multiple strategies), so
    this writes the safetensors/manifest files directly to construct
    a cache shape that is a valid (if not currently produced) input
    to the pipeline: multiple pool strategies sharing one cache
    directory.
    """
    cache_dir = tmp_path / "multi_pool_cache"
    cache_dir.mkdir()

    rng = np.random.default_rng(2)
    manifest_entries: list[dict[str, Any]] = []
    tensors_by_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}
    example_id = 0

    for pool_strategy in ("last_token", "mean"):
        y = [0] * (n_per_group // 2) + [1] * (n_per_group // 2)
        y_arr = np.array(y)
        for layer in layer_indices:
            separation = layer * 2.0
            X = rng.normal(size=(n_per_group, hidden)).astype(np.float32)
            X[:, 0] += separation * y_arr
            tensors_by_layer[layer].append(torch.tensor(X))
        for i in range(n_per_group):
            manifest_entries.append(
                {
                    "example_id": example_id,
                    "label": y[i],
                    "token_count": 5,
                    "layer_indices": list(layer_indices),
                    "pool_strategy": pool_strategy,
                    "model_key": "mock-model",
                    "model_revision": "main",
                    "dataset_name": "sst2",
                    "split": "validation",
                    "seed": 42,
                    "torch_version": torch.__version__,
                }
            )
            example_id += 1

    for layer in layer_indices:
        tensor = torch.cat(tensors_by_layer[layer], dim=0)
        save_file(
            {"representations": tensor.half().contiguous()},
            str(cache_dir / f"layer_{layer:02d}.safetensors"),
        )

    with open(cache_dir / "manifest.jsonl", "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    return cache_dir


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


# ---------------------------------------------------------------------------
# TestStackPredictionDepths
# ---------------------------------------------------------------------------


class TestStackPredictionDepths:
    """Tests for the per-layer-prediction stacking helper."""

    def test_matches_manual_prediction_depth_per_example(self) -> None:
        """Depths should equal prediction_depth applied to each example's
        own sequence of per-layer predictions, in layer order."""
        layer_predictions = {
            0: np.array([0, 1, 0]),
            1: np.array([0, 0, 0]),
            2: np.array([1, 0, 0]),
        }
        layer_indices = [0, 1, 2]

        depths = _stack_prediction_depths(layer_predictions, layer_indices)

        expected = [
            prediction_depth([0, 0, 1]),  # example 0
            prediction_depth([1, 0, 0]),  # example 1
            prediction_depth([0, 0, 0]),  # example 2
        ]
        np.testing.assert_array_equal(depths, expected)

    def test_respects_layer_index_order_not_dict_insertion_order(self) -> None:
        """The sequence passed to prediction_depth must follow
        layer_indices, even if the dict was populated out of order."""
        layer_predictions = {
            2: np.array([1]),
            0: np.array([1]),
            1: np.array([0]),
        }
        depths = _stack_prediction_depths(layer_predictions, [0, 1, 2])
        assert depths[0] == prediction_depth([1, 0, 1])

    def test_returns_one_depth_per_example(self) -> None:
        """Output length should match the number of examples, not the
        number of layers."""
        layer_predictions = {0: np.zeros(7), 1: np.zeros(7)}
        depths = _stack_prediction_depths(layer_predictions, [0, 1])
        assert depths.shape == (7,)


# ---------------------------------------------------------------------------
# TestProbingPipelineOutput
# ---------------------------------------------------------------------------


class TestProbingPipelineOutput:
    """Tests for CSV persistence of the tidy result table."""

    def test_writes_csv_to_expected_path(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """The output file should follow {output_dir}/{model}_{task}.csv."""
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        pipeline.run(classification_cache_dir)

        expected_path = Path(probing_config["output_dir"]) / "mock-model_sst2.csv"
        assert expected_path.exists()

    def test_creates_output_directory_if_missing(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """output_dir need not exist beforehand."""
        assert not Path(probing_config["output_dir"]).exists()
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        pipeline.run(classification_cache_dir)
        assert Path(probing_config["output_dir"]).exists()

    def test_csv_contents_match_returned_rows(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """Reading the written CSV back should reproduce the same rows
        (as strings) that run() returned."""
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(classification_cache_dir)

        csv_path = Path(probing_config["output_dir"]) / "mock-model_sst2.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        assert len(csv_rows) == len(rows)
        for csv_row, row in zip(csv_rows, rows):
            for column in RESULT_COLUMNS:
                assert csv_row[column] == str(row[column])

    def test_csv_header_matches_result_columns(
        self, classification_cache_dir: Path, probing_config: dict[str, Any]
    ) -> None:
        """The CSV header row should exactly match RESULT_COLUMNS, in
        order."""
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        pipeline.run(classification_cache_dir)

        csv_path = Path(probing_config["output_dir"]) / "mock-model_sst2.csv"
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f))
        assert header == RESULT_COLUMNS


# ---------------------------------------------------------------------------
# TestProbingPipelineMultiPoolStrategy
# ---------------------------------------------------------------------------


class TestProbingPipelineMultiPoolStrategy:
    """Integration tests for a cache directory holding multiple pool
    strategies, exercising the (layer, pool_strategy) iteration in full."""

    def test_row_count_covers_every_combination(
        self, tmp_path: Path, probing_config: dict[str, Any]
    ) -> None:
        """Two pool strategies x two layers should produce four rows."""
        cache_dir = _make_multi_pool_strategy_cache(tmp_path, layer_indices=(0, 1))
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(cache_dir)
        assert len(rows) == 4

    def test_both_pool_strategies_present(
        self, tmp_path: Path, probing_config: dict[str, Any]
    ) -> None:
        """Both pool_strategy values from the manifest should appear in
        the result table."""
        cache_dir = _make_multi_pool_strategy_cache(tmp_path, layer_indices=(0, 1))
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(cache_dir)
        assert {row["pool_strategy"] for row in rows} == {"last_token", "mean"}

    def test_prediction_depth_mean_constant_within_each_pool_strategy(
        self, tmp_path: Path, probing_config: dict[str, Any]
    ) -> None:
        """prediction_depth_mean is a whole-trajectory summary, so every
        layer row for a given pool strategy should carry the same
        value, independent of the other pool strategy's value."""
        cache_dir = _make_multi_pool_strategy_cache(tmp_path, layer_indices=(0, 1))
        pipeline = ProbingPipeline("mock-model", "sst2", probing_config)
        rows = pipeline.run(cache_dir)

        by_strategy: dict[str, set[float]] = {}
        for row in rows:
            by_strategy.setdefault(row["pool_strategy"], set()).add(
                row["prediction_depth_mean"]
            )

        for depth_values in by_strategy.values():
            assert len(depth_values) == 1

    def test_pool_strategies_do_not_leak_examples_into_each_other(
        self, tmp_path: Path, probing_config: dict[str, Any]
    ) -> None:
        """Each pool strategy's probe should only ever see its own
        examples: swapping which strategy is queried first should not
        change either strategy's score."""
        cache_dir = _make_multi_pool_strategy_cache(tmp_path, layer_indices=(0, 1))
        pipeline_a = ProbingPipeline("mock-model", "sst2", probing_config)
        rows_a = pipeline_a.run(cache_dir)

        pipeline_b = ProbingPipeline("mock-model", "sst2", probing_config)
        rows_b = pipeline_b.run(cache_dir)

        scores_a = {(r["pool_strategy"], r["layer"]): r["score"] for r in rows_a}
        scores_b = {(r["pool_strategy"], r["layer"]): r["score"] for r in rows_b}
        assert scores_a == scores_b
