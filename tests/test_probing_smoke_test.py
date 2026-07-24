"""
Tests for src.probing.smoke_test.

All tests are fully mocked or run against small synthetic/real (but
tiny) data structures — no network access, model downloads, or GPU
required. Covers the LAMA/T-REx subset-selection logic, the manual
hook-based extraction helper, the per-example prediction-depth
recomputation helper, and every validation check function in both its
pass and fail paths.
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from torch import nn

from src.data.tasks import TaskSpec
from src.extraction.cache import save_representations
from src.probing.plotting import PlotArtifact
from src.probing.smoke_test import (
    _check_cache_alignment,
    _check_layerwise_training,
    _check_metrics_populated,
    _check_plots_written,
    _check_qualitative_trend,
    _check_selectivity_above_threshold,
    _compute_example_depth_rows,
    _extract_task_representations,
    _load_validation_dataset,
    _select_lama_trex_subset,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOCK_NUM_LAYERS = 3
MOCK_HIDDEN_SIZE = 16
MOCK_SEQ_LEN = 6


# ---------------------------------------------------------------------------
# Fake GPTNeoX-shaped model for extraction testing (mirrors the pattern
# used in tests/test_pipeline.py for the extraction pipeline).
# ---------------------------------------------------------------------------


class _FakeEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        batch, seq = input_ids.shape
        return torch.randn(batch, seq, self._hidden_size)


class _FakeGPTNeoXLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        return (torch.randn_like(hidden_states),)


class _FakeGPTNeoXModel(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_in = _FakeEmbedding(hidden_size)
        self.layers = nn.ModuleList(
            [_FakeGPTNeoXLayer(hidden_size) for _ in range(num_layers)]
        )


class _FakeModel(nn.Module):
    """Minimal GPTNeoX-shaped model, real enough for HookManager to hook into."""

    def __init__(
        self, num_layers: int = MOCK_NUM_LAYERS, hidden_size: int = MOCK_HIDDEN_SIZE
    ) -> None:
        super().__init__()
        self.gpt_neox = _FakeGPTNeoXModel(num_layers, hidden_size)

    def forward(
        self, input_ids: torch.Tensor | None = None, attention_mask: Any = None, **kwargs: Any
    ) -> torch.Tensor:
        hidden = self.gpt_neox.embed_in(input_ids)
        for layer in self.gpt_neox.layers:
            hidden = layer(hidden)[0]
        return hidden


def _make_fake_tokenizer() -> MagicMock:
    """Create a tokenizer mock that returns real tensors for any batch."""
    tokenizer = MagicMock()

    def tokenize_fn(texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.randint(0, 100, (batch_size, MOCK_SEQ_LEN)),
            "attention_mask": torch.ones(batch_size, MOCK_SEQ_LEN, dtype=torch.long),
        }

    tokenizer.side_effect = tokenize_fn
    return tokenizer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_task_spec() -> TaskSpec:
    """A mock binary classification task, independent of tasks.yaml."""
    return TaskSpec(
        name="mock_cls",
        family="shallow_semantic",
        label_type="classification",
        num_classes=2,
        extraction_position="last_token",
        description="mock classification task",
    )


@pytest.fixture
def regression_task_spec() -> TaskSpec:
    """A mock regression task, independent of tasks.yaml."""
    return TaskSpec(
        name="mock_reg",
        family="factual_lookup",
        label_type="regression",
        num_classes=None,
        extraction_position="last_token",
        description="mock regression task",
    )


@pytest.fixture
def probing_config(tmp_path: Path) -> dict[str, Any]:
    """A minimal in-memory probing config with output routed to tmp_path."""
    return {
        "probe": {
            "classification": "logistic_regression",
            "regression": "ridge",
            "standardize": True,
            "class_weight": "balanced",
            "regularization_grid": [0.1, 1.0],
        },
        "cv": {"folds": 5, "seed": 42},
        "output_dir": str(tmp_path / "probing_output"),
    }


def _make_lama_pool(counts: dict[str, int]) -> Dataset:
    """Build a synthetic LAMA/T-REx-shaped pool with a given relation mix."""
    records: list[dict[str, Any]] = []
    example_id = 0
    for relation, n in counts.items():
        for i in range(n):
            records.append(
                {
                    "text": f"{relation} example {i}",
                    "label": abs(hash(relation)) % 1000,
                    "example_id": example_id,
                    "obj_label": "obj",
                    "sub_label": "sub",
                    "predicate_id": relation,
                }
            )
            example_id += 1
    columns = {key: [r[key] for r in records] for key in records[0]}
    return Dataset.from_dict(columns)


def _make_classification_cache(tmp_path: Path, n: int = 30, num_layers: int = 2) -> Path:
    """Build a real cache directory with layer-separable binary-label data."""
    rng = np.random.default_rng(0)
    y = [0] * (n // 2) + [1] * (n // 2)
    y_arr = np.array(y)

    representations: dict[int, torch.Tensor] = {}
    for layer in range(num_layers):
        separation = layer * 2.0
        X = rng.normal(size=(n, 4)).astype(np.float32)
        X[:, 0] += separation * y_arr
        representations[layer] = torch.tensor(X)

    return save_representations(
        representations=representations,
        labels=y,
        example_ids=list(range(n)),
        token_counts=[5] * n,
        output_dir=str(tmp_path / "cache"),
        model_key="mock-model",
        dataset_name="mock_cls",
        split="validation",
        pool_strategy="last_token",
        seed=42,
    )


# ---------------------------------------------------------------------------
# TestSelectLamaTrexSubset
# ---------------------------------------------------------------------------


class TestSelectLamaTrexSubset:
    """Tests for the LAMA/T-REx relation-frequency subset selector."""

    @patch("src.probing.smoke_test.load_lama_trex")
    def test_keeps_only_top_relations(self, mock_load: MagicMock) -> None:
        """Only the num_relations most frequent relations should survive."""
        mock_load.return_value = _make_lama_pool({"P19": 50, "P20": 30, "P37": 15, "P999": 5})

        subset = _select_lama_trex_subset(
            pool_size=100, num_relations=2, subset_size=1000, seed=1
        )

        assert set(subset["predicate_id"]) == {"P19", "P20"}

    @patch("src.probing.smoke_test.load_lama_trex")
    def test_returns_whole_filtered_pool_when_smaller_than_subset_size(
        self, mock_load: MagicMock
    ) -> None:
        """If the filtered pool is smaller than subset_size, return it whole."""
        mock_load.return_value = _make_lama_pool({"P19": 50, "P20": 30, "P37": 15})

        subset = _select_lama_trex_subset(
            pool_size=100, num_relations=2, subset_size=1000, seed=1
        )

        assert len(subset) == 80  # 50 + 30

    @patch("src.probing.smoke_test.load_lama_trex")
    def test_subsamples_down_to_requested_size(self, mock_load: MagicMock) -> None:
        """When the filtered pool exceeds subset_size, it is subsampled."""
        mock_load.return_value = _make_lama_pool({"P19": 50, "P20": 30, "P37": 15})

        subset = _select_lama_trex_subset(pool_size=100, num_relations=3, subset_size=20, seed=1)

        assert len(subset) == 20
        # Subsampling must not introduce relations outside the filtered set.
        assert set(subset["predicate_id"]) <= {"P19", "P20", "P37"}

    @patch("src.probing.smoke_test.load_lama_trex")
    def test_deterministic_given_seed(self, mock_load: MagicMock) -> None:
        """Same seed should produce the same subsample."""
        mock_load.return_value = _make_lama_pool({"P19": 50, "P20": 30})

        subset_a = _select_lama_trex_subset(pool_size=80, num_relations=2, subset_size=10, seed=7)
        subset_b = _select_lama_trex_subset(pool_size=80, num_relations=2, subset_size=10, seed=7)

        assert list(subset_a["example_id"]) == list(subset_b["example_id"])


# ---------------------------------------------------------------------------
# TestLoadValidationDataset
# ---------------------------------------------------------------------------


class TestLoadValidationDataset:
    """Tests for the per-task validation dataset dispatcher."""

    @patch("src.probing.smoke_test.load_sst2")
    def test_dispatches_sst2_to_load_sst2(self, mock_load_sst2: MagicMock) -> None:
        """sst2 should be routed to load_sst2 with the requested size and seed."""
        mock_load_sst2.return_value = "sst2-dataset"

        result = _load_validation_dataset("sst2", max_examples=32, seed=5)

        assert result == "sst2-dataset"
        mock_load_sst2.assert_called_once_with(split="validation", max_examples=32, seed=5)

    @patch("src.probing.smoke_test._select_lama_trex_subset")
    def test_dispatches_lama_trex_to_subset_selector(self, mock_select: MagicMock) -> None:
        """lama_trex should be routed to _select_lama_trex_subset."""
        mock_select.return_value = "lama-dataset"

        result = _load_validation_dataset("lama_trex", max_examples=40, seed=5)

        assert result == "lama-dataset"
        mock_select.assert_called_once()

    def test_unknown_task_raises(self) -> None:
        """An unregistered task name should raise a clear error."""
        with pytest.raises(ValueError, match="No validation dataset loader"):
            _load_validation_dataset("unknown_task", max_examples=10, seed=1)


# ---------------------------------------------------------------------------
# TestExtractTaskRepresentations
# ---------------------------------------------------------------------------


class TestExtractTaskRepresentations:
    """Tests for the manual hook-based extraction helper."""

    def test_cache_contains_every_layer(self, tmp_path: Path) -> None:
        """The cache should have one safetensors file per hidden-state layer."""
        model = _FakeModel()
        tokenizer = _make_fake_tokenizer()
        dataset = Dataset.from_dict(
            {
                "text": [f"example {i}" for i in range(5)],
                "label": [0, 1, 0, 1, 0],
                "example_id": list(range(5)),
            }
        )

        cache_dir = _extract_task_representations(
            model,
            tokenizer,
            dataset,
            model_key="mock-model",
            task_name="sst2",
            pool_strategy="last_token",
            split="validation",
            output_dir=tmp_path,
            seed=42,
            model_revision="main",
            batch_size=2,
        )

        safetensor_files = list(cache_dir.glob("layer_*.safetensors"))
        assert len(safetensor_files) == MOCK_NUM_LAYERS + 1

    def test_manifest_matches_dataset(self, tmp_path: Path) -> None:
        """Manifest labels/example_ids should match the source dataset exactly."""
        model = _FakeModel()
        tokenizer = _make_fake_tokenizer()
        dataset = Dataset.from_dict(
            {
                "text": [f"example {i}" for i in range(5)],
                "label": [0, 1, 0, 1, 0],
                "example_id": [10, 11, 12, 13, 14],
            }
        )

        cache_dir = _extract_task_representations(
            model,
            tokenizer,
            dataset,
            model_key="mock-model",
            task_name="sst2",
            pool_strategy="last_token",
            split="validation",
            output_dir=tmp_path,
            seed=42,
            model_revision="main",
            batch_size=2,
        )

        from src.extraction.cache import load_representations

        _, manifest = load_representations(cache_dir)
        assert [entry["label"] for entry in manifest] == [0, 1, 0, 1, 0]
        assert [entry["example_id"] for entry in manifest] == [10, 11, 12, 13, 14]

    def test_handles_batch_size_not_dividing_dataset_evenly(self, tmp_path: Path) -> None:
        """5 examples with batch_size=2 (uneven final batch) should still work."""
        model = _FakeModel()
        tokenizer = _make_fake_tokenizer()
        dataset = Dataset.from_dict(
            {
                "text": [f"example {i}" for i in range(5)],
                "label": [0, 1, 0, 1, 0],
                "example_id": list(range(5)),
            }
        )

        cache_dir = _extract_task_representations(
            model,
            tokenizer,
            dataset,
            model_key="mock-model",
            task_name="sst2",
            pool_strategy="last_token",
            split="validation",
            output_dir=tmp_path,
            seed=42,
            model_revision="main",
            batch_size=2,
        )

        from src.extraction.cache import load_representations

        representations, manifest = load_representations(cache_dir)
        assert len(manifest) == 5
        for tensor in representations.values():
            assert tensor.shape[0] == 5


# ---------------------------------------------------------------------------
# TestComputeExampleDepthRows
# ---------------------------------------------------------------------------


class TestComputeExampleDepthRows:
    """Tests for the per-example prediction-depth recomputation helper."""

    def test_returns_one_row_per_example(
        self, tmp_path: Path, classification_task_spec: TaskSpec, probing_config: dict[str, Any]
    ) -> None:
        """Row count should match the number of cached examples."""
        cache_dir = _make_classification_cache(tmp_path, n=30, num_layers=2)

        rows = _compute_example_depth_rows(
            cache_dir, classification_task_spec, probing_config, "mock_cls"
        )

        assert len(rows) == 30

    def test_rows_have_expected_keys(
        self, tmp_path: Path, classification_task_spec: TaskSpec, probing_config: dict[str, Any]
    ) -> None:
        """Each row should carry task, pool_strategy, example_id, prediction_depth."""
        cache_dir = _make_classification_cache(tmp_path, n=20, num_layers=2)

        rows = _compute_example_depth_rows(
            cache_dir, classification_task_spec, probing_config, "mock_cls"
        )

        for row in rows:
            assert set(row.keys()) == {"task", "pool_strategy", "example_id", "prediction_depth"}
            assert row["task"] == "mock_cls"
            assert row["pool_strategy"] == "last_token"

    def test_prediction_depth_within_layer_range(
        self, tmp_path: Path, classification_task_spec: TaskSpec, probing_config: dict[str, Any]
    ) -> None:
        """Every prediction depth should be a valid layer index."""
        cache_dir = _make_classification_cache(tmp_path, n=20, num_layers=3)

        rows = _compute_example_depth_rows(
            cache_dir, classification_task_spec, probing_config, "mock_cls"
        )

        for row in rows:
            assert 0 <= row["prediction_depth"] <= 2


# ---------------------------------------------------------------------------
# TestCheckCacheAlignment
# ---------------------------------------------------------------------------


class TestCheckCacheAlignment:
    """Tests for the cache/dataset alignment check."""

    def test_passes_for_aligned_cache(self, tmp_path: Path) -> None:
        """A cache built directly from a dataset should pass without error."""
        n = 10
        dataset = Dataset.from_dict(
            {
                "text": [f"ex {i}" for i in range(n)],
                "label": [i % 2 for i in range(n)],
                "example_id": list(range(n)),
            }
        )
        cache_dir = save_representations(
            representations={0: torch.randn(n, 4)},
            labels=list(dataset["label"]),
            example_ids=list(dataset["example_id"]),
            token_counts=[5] * n,
            output_dir=str(tmp_path / "cache"),
            model_key="mock-model",
            dataset_name="sst2",
            split="validation",
            pool_strategy="last_token",
            seed=42,
        )

        _check_cache_alignment(cache_dir, dataset)  # should not raise

    def test_raises_on_label_mismatch(self, tmp_path: Path) -> None:
        """A cache with labels that don't match the dataset should raise."""
        n = 10
        dataset = Dataset.from_dict(
            {
                "text": [f"ex {i}" for i in range(n)],
                "label": [0] * n,
                "example_id": list(range(n)),
            }
        )
        cache_dir = save_representations(
            representations={0: torch.randn(n, 4)},
            labels=[1] * n,  # deliberately wrong
            example_ids=list(dataset["example_id"]),
            token_counts=[5] * n,
            output_dir=str(tmp_path / "cache"),
            model_key="mock-model",
            dataset_name="sst2",
            split="validation",
            pool_strategy="last_token",
            seed=42,
        )

        with pytest.raises(AssertionError, match="labels do not match"):
            _check_cache_alignment(cache_dir, dataset)

    def test_raises_on_length_mismatch(self, tmp_path: Path) -> None:
        """A cache with a different example count than the dataset should raise."""
        n = 10
        dataset = Dataset.from_dict(
            {
                "text": [f"ex {i}" for i in range(n)],
                "label": [0] * n,
                "example_id": list(range(n)),
            }
        )
        cache_dir = save_representations(
            representations={0: torch.randn(n - 1, 4)},
            labels=[0] * (n - 1),
            example_ids=list(range(n - 1)),
            token_counts=[5] * (n - 1),
            output_dir=str(tmp_path / "cache"),
            model_key="mock-model",
            dataset_name="sst2",
            split="validation",
            pool_strategy="last_token",
            seed=42,
        )

        with pytest.raises(AssertionError, match="Manifest has"):
            _check_cache_alignment(cache_dir, dataset)


# ---------------------------------------------------------------------------
# TestCheckLayerwiseTraining
# ---------------------------------------------------------------------------


class TestCheckLayerwiseTraining:
    """Tests for the layerwise-training-succeeded check."""

    def test_passes_when_every_layer_covered_with_finite_scores(self) -> None:
        """All expected layers present with finite scores should not raise."""
        rows = [{"layer": layer, "score": 0.8} for layer in range(4)]
        _check_layerwise_training(rows, expected_layers={0, 1, 2, 3})  # should not raise

    def test_raises_on_missing_layer(self) -> None:
        """A missing expected layer should raise."""
        rows = [{"layer": layer, "score": 0.8} for layer in range(3)]
        with pytest.raises(AssertionError, match="do not match"):
            _check_layerwise_training(rows, expected_layers={0, 1, 2, 3})

    def test_raises_on_nonfinite_score(self) -> None:
        """A NaN score should raise even if every layer is covered."""
        rows = [{"layer": 0, "score": 0.8}, {"layer": 1, "score": float("nan")}]
        with pytest.raises(AssertionError, match="Non-finite score"):
            _check_layerwise_training(rows, expected_layers={0, 1})

    def test_raises_on_empty_rows(self) -> None:
        """Empty rows should raise rather than vacuously pass."""
        with pytest.raises(AssertionError, match="no rows"):
            _check_layerwise_training([], expected_layers={0})


# ---------------------------------------------------------------------------
# TestCheckSelectivityAboveThreshold
# ---------------------------------------------------------------------------


class TestCheckSelectivityAboveThreshold:
    """Tests for the mid-layer selectivity check."""

    def test_passes_when_mid_layer_selectivity_high(self) -> None:
        """A high selectivity at a mid-depth layer should not raise."""
        rows = [{"layer": layer, "selectivity": 0.01 * layer} for layer in range(7)]
        rows[3]["selectivity"] = 0.5  # mid-layer (of 0..6) is well within range
        _check_selectivity_above_threshold(rows, threshold=0.05)  # should not raise

    def test_raises_when_all_selectivity_near_zero(self) -> None:
        """Selectivity near zero everywhere should raise."""
        rows = [{"layer": layer, "selectivity": 0.001} for layer in range(7)]
        with pytest.raises(AssertionError, match="did not exceed"):
            _check_selectivity_above_threshold(rows, threshold=0.05)

    def test_ignores_high_selectivity_outside_mid_range(self) -> None:
        """A high selectivity only at the very first/last layer should not count."""
        rows = [{"layer": layer, "selectivity": 0.001} for layer in range(7)]
        rows[0]["selectivity"] = 0.9  # shallowest layer, outside (0.25, 0.75) range
        rows[6]["selectivity"] = 0.9  # deepest layer, outside range
        with pytest.raises(AssertionError, match="did not exceed"):
            _check_selectivity_above_threshold(rows, threshold=0.05)


# ---------------------------------------------------------------------------
# TestCheckMetricsPopulated
# ---------------------------------------------------------------------------


class TestCheckMetricsPopulated:
    """Tests for the margin/ECE/prediction-depth population check."""

    def test_passes_for_classification_with_finite_metrics(
        self, classification_task_spec: TaskSpec
    ) -> None:
        """Finite margin/ece/prediction_depth_mean should not raise for classification."""
        rows = [
            {"layer": 0, "margin_mean": 1.0, "ece": 0.1, "prediction_depth_mean": 0.5},
        ]
        _check_metrics_populated(rows, classification_task_spec)  # should not raise

    def test_raises_on_nan_margin_for_classification(
        self, classification_task_spec: TaskSpec
    ) -> None:
        """A NaN margin_mean should raise for a classification task."""
        rows = [
            {"layer": 0, "margin_mean": float("nan"), "ece": 0.1, "prediction_depth_mean": 0.5},
        ]
        with pytest.raises(AssertionError, match="margin_mean"):
            _check_metrics_populated(rows, classification_task_spec)

    def test_allows_nan_margin_for_regression(self, regression_task_spec: TaskSpec) -> None:
        """NaN margin/ece is expected (and allowed) for regression tasks."""
        rows = [
            {
                "layer": 0,
                "margin_mean": float("nan"),
                "ece": float("nan"),
                "prediction_depth_mean": 0.5,
            },
        ]
        _check_metrics_populated(rows, regression_task_spec)  # should not raise

    def test_raises_on_nan_prediction_depth_regardless_of_task_type(
        self, regression_task_spec: TaskSpec
    ) -> None:
        """prediction_depth_mean must be finite even for regression tasks."""
        rows = [
            {
                "layer": 0,
                "margin_mean": float("nan"),
                "ece": float("nan"),
                "prediction_depth_mean": float("nan"),
            },
        ]
        with pytest.raises(AssertionError, match="prediction_depth_mean"):
            _check_metrics_populated(rows, regression_task_spec)


# ---------------------------------------------------------------------------
# TestCheckPlotsWritten
# ---------------------------------------------------------------------------


class TestCheckPlotsWritten:
    """Tests for the plot-output existence/validity check."""

    _PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

    def _write_fake_png(self, path: Path, size: int = 200) -> None:
        path.write_bytes(self._PNG_MAGIC + b"\x00" * size)

    def test_passes_for_valid_artifacts(self, tmp_path: Path) -> None:
        """Two well-formed PNG/CSV artifact pairs should not raise."""
        score_png = tmp_path / "score.png"
        score_csv = tmp_path / "score.csv"
        depth_png = tmp_path / "depth.png"
        depth_csv = tmp_path / "depth.csv"

        self._write_fake_png(score_png)
        score_csv.write_text("a,b\n1,2\n")
        self._write_fake_png(depth_png)
        depth_csv.write_text("a,b\n1,2\n")

        _check_plots_written(
            PlotArtifact(figure_path=score_png, data_path=score_csv),
            PlotArtifact(figure_path=depth_png, data_path=depth_csv),
        )  # should not raise

    def test_raises_on_missing_figure(self, tmp_path: Path) -> None:
        """A figure path that doesn't exist should raise."""
        score_csv = tmp_path / "score.csv"
        score_csv.write_text("a,b\n1,2\n")
        depth_png = tmp_path / "depth.png"
        depth_csv = tmp_path / "depth.csv"
        self._write_fake_png(depth_png)
        depth_csv.write_text("a,b\n1,2\n")

        with pytest.raises(AssertionError, match="Missing figure"):
            _check_plots_written(
                PlotArtifact(figure_path=tmp_path / "missing.png", data_path=score_csv),
                PlotArtifact(figure_path=depth_png, data_path=depth_csv),
            )

    def test_raises_on_invalid_png_header(self, tmp_path: Path) -> None:
        """A file that isn't actually a PNG should raise."""
        bad_png = tmp_path / "bad.png"
        bad_png.write_bytes(b"not a png" + b"\x00" * 200)
        score_csv = tmp_path / "score.csv"
        score_csv.write_text("a,b\n1,2\n")
        depth_png = tmp_path / "depth.png"
        depth_csv = tmp_path / "depth.csv"
        self._write_fake_png(depth_png)
        depth_csv.write_text("a,b\n1,2\n")

        with pytest.raises(AssertionError, match="not a valid PNG"):
            _check_plots_written(
                PlotArtifact(figure_path=bad_png, data_path=score_csv),
                PlotArtifact(figure_path=depth_png, data_path=depth_csv),
            )

    def test_raises_on_empty_companion_csv(self, tmp_path: Path) -> None:
        """An empty companion CSV should raise even if the PNG is valid."""
        score_png = tmp_path / "score.png"
        self._write_fake_png(score_png)
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        depth_png = tmp_path / "depth.png"
        depth_csv = tmp_path / "depth.csv"
        self._write_fake_png(depth_png)
        depth_csv.write_text("a,b\n1,2\n")

        with pytest.raises(AssertionError, match="is empty"):
            _check_plots_written(
                PlotArtifact(figure_path=score_png, data_path=empty_csv),
                PlotArtifact(figure_path=depth_png, data_path=depth_csv),
            )


# ---------------------------------------------------------------------------
# TestCheckQualitativeTrend
# ---------------------------------------------------------------------------


class TestCheckQualitativeTrend:
    """Tests for the soft qualitative rising-accuracy sanity check."""

    def test_reports_true_when_score_rises(self, caplog: pytest.LogCaptureFixture) -> None:
        """A monotonically rising score should report rises_by_mid_depth=True."""
        rows = [{"layer": layer, "score": 0.5 + 0.05 * layer} for layer in range(8)]

        with caplog.at_level(logging.INFO, logger="src.probing.smoke_test"):
            result = _check_qualitative_trend(rows)

        assert result["rises_by_mid_depth"] is True
        assert result["late_mean_score"] > result["early_mean_score"]
        assert "PASSED" in caplog.text

    def test_reports_false_and_warns_when_score_falls(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A falling score should report rises_by_mid_depth=False and log a warning."""
        rows = [{"layer": layer, "score": 0.9 - 0.05 * layer} for layer in range(8)]

        with caplog.at_level(logging.WARNING, logger="src.probing.smoke_test"):
            result = _check_qualitative_trend(rows)

        assert result["rises_by_mid_depth"] is False
        assert "not observed" in caplog.text

    def test_does_not_raise_on_mismatch(self) -> None:
        """A mismatched trend must not raise — this is a soft, logged check only."""
        rows = [{"layer": layer, "score": 0.9 - 0.05 * layer} for layer in range(8)]
        _check_qualitative_trend(rows)  # should not raise
