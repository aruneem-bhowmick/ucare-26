"""
Tests for probing plot utilities.

Validates that both plotting functions filter to the requested task,
group by pool strategy correctly, write a non-trivial PNG figure and a
companion CSV that reproduces exactly the data that was plotted, and
raise clear errors on malformed input. Uses synthetic tidy-table rows
throughout — no real cache, model, or probe I/O.
"""

import csv
from pathlib import Path
from typing import Any

import matplotlib
import pytest

from src.probing.plotting import (
    PlotArtifact,
    plot_prediction_depth_distribution,
    plot_score_vs_layer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def score_results() -> list[dict[str, Any]]:
    """Synthetic per-layer summary rows for two tasks and two pool
    strategies, with layer order deliberately shuffled."""
    rows: list[dict[str, Any]] = []
    for task in ("sst2", "lama_trex"):
        for pool_strategy in ("last_token", "mean"):
            offset = 0.1 if pool_strategy == "mean" else 0.0
            for layer in (2, 0, 1):
                rows.append(
                    {
                        "model": "mock-model",
                        "task": task,
                        "pool_strategy": pool_strategy,
                        "layer": layer,
                        "score": 0.5 + 0.1 * layer + offset,
                        "score_std": 0.02,
                        "selectivity": 0.3 + 0.05 * layer,
                        "margin_mean": 1.0 + 0.2 * layer,
                        "ece": 0.1,
                        "prediction_depth_mean": 1.5,
                    }
                )
    return rows


@pytest.fixture
def depth_results() -> list[dict[str, Any]]:
    """Synthetic per-example prediction-depth rows for two tasks and two
    pool strategies."""
    rows: list[dict[str, Any]] = []
    example_id = 0
    for task in ("sst2", "lama_trex"):
        for pool_strategy in ("last_token", "mean"):
            for depth in (0, 1, 1, 2, 3, 3, 3):
                rows.append(
                    {
                        "task": task,
                        "pool_strategy": pool_strategy,
                        "example_id": example_id,
                        "prediction_depth": depth,
                    }
                )
                example_id += 1
    return rows


# ---------------------------------------------------------------------------
# TestHeadlessBackend
# ---------------------------------------------------------------------------


class TestHeadlessBackend:
    """Tests that the plotting module forces a headless backend."""

    def test_backend_is_agg(self) -> None:
        """Importing src.probing.plotting should switch matplotlib to
        the non-interactive Agg backend."""
        assert matplotlib.get_backend().lower() == "agg"


# ---------------------------------------------------------------------------
# TestPlotScoreVsLayer
# ---------------------------------------------------------------------------


class TestPlotScoreVsLayer:
    """Tests for the per-layer metric plotting function."""

    def test_returns_plot_artifact_with_existing_files(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """The figure and companion CSV should both exist at the
        returned paths."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", output_dir=tmp_path
        )
        assert isinstance(artifact, PlotArtifact)
        assert artifact.figure_path.exists()
        assert artifact.data_path.exists()

    def test_figure_and_csv_paths_follow_naming_convention(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Output filenames should follow {task}_{metric}_vs_layer.{ext}."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", metric="selectivity", output_dir=tmp_path
        )
        assert artifact.figure_path == tmp_path / "sst2_selectivity_vs_layer.png"
        assert artifact.data_path == tmp_path / "sst2_selectivity_vs_layer.csv"

    def test_figure_file_is_non_empty_png(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """The saved figure should be a non-trivial PNG file."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", output_dir=tmp_path
        )
        data = artifact.figure_path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(data) > 100

    def test_filters_to_requested_task(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Rows belonging to other tasks should not appear in the
        companion CSV."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            csv_rows = list(csv.DictReader(f))
        assert {row["task"] for row in csv_rows} == {"sst2"}

    def test_both_pool_strategies_present(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Every pool_strategy present for the task should appear in
        the companion CSV."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            csv_rows = list(csv.DictReader(f))
        assert {row["pool_strategy"] for row in csv_rows} == {
            "last_token",
            "mean",
        }

    def test_rows_sorted_by_layer_within_each_pool_strategy(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Even though the input layer order is shuffled, each pool
        strategy's rows in the CSV should be in ascending layer order."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            csv_rows = list(csv.DictReader(f))

        by_strategy: dict[str, list[int]] = {}
        for row in csv_rows:
            by_strategy.setdefault(row["pool_strategy"], []).append(
                int(row["layer"])
            )
        for layers in by_strategy.values():
            assert layers == sorted(layers)

    def test_score_metric_includes_std_column(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """metric='score' has a companion score_std column in
        RESULT_COLUMNS, so the CSV should include it."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", metric="score", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            header = next(csv.reader(f))
        assert header == ["task", "pool_strategy", "layer", "score", "score_std"]

    def test_metric_without_std_omits_std_column(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """selectivity has no companion _std column, so the CSV should
        not include one."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", metric="selectivity", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            header = next(csv.reader(f))
        assert header == ["task", "pool_strategy", "layer", "selectivity"]

    def test_csv_values_match_input_rows(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Each plotted CSV row's score should match the corresponding
        input row for that (pool_strategy, layer)."""
        artifact = plot_score_vs_layer(
            score_results, "sst2", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            csv_rows = list(csv.DictReader(f))

        expected = {
            (row["pool_strategy"], row["layer"]): row["score"]
            for row in score_results
            if row["task"] == "sst2"
        }
        for row in csv_rows:
            key = (row["pool_strategy"], int(row["layer"]))
            assert float(row["score"]) == pytest.approx(expected[key])

    def test_creates_output_directory_if_missing(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """output_dir need not exist beforehand."""
        output_dir = tmp_path / "nested" / "plots"
        assert not output_dir.exists()
        plot_score_vs_layer(score_results, "sst2", output_dir=output_dir)
        assert output_dir.exists()

    def test_raises_on_unknown_task(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """A task_name with no matching rows should raise a clear
        error."""
        with pytest.raises(ValueError, match="No rows found for task"):
            plot_score_vs_layer(score_results, "unknown_task", output_dir=tmp_path)

    def test_raises_on_unknown_metric(
        self, score_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """A metric not present in the result rows should raise a clear
        error."""
        with pytest.raises(ValueError, match="metric 'not_a_column'"):
            plot_score_vs_layer(
                score_results, "sst2", metric="not_a_column", output_dir=tmp_path
            )


# ---------------------------------------------------------------------------
# TestPlotPredictionDepthDistribution
# ---------------------------------------------------------------------------


class TestPlotPredictionDepthDistribution:
    """Tests for the per-example prediction-depth histogram function."""

    def test_returns_plot_artifact_with_existing_files(
        self, depth_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """The figure and companion CSV should both exist at the
        returned paths."""
        artifact = plot_prediction_depth_distribution(
            depth_results, "sst2", output_dir=tmp_path
        )
        assert artifact.figure_path.exists()
        assert artifact.data_path.exists()

    def test_figure_and_csv_paths_follow_naming_convention(
        self, depth_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Output filenames should follow
        {task}_prediction_depth_distribution.{ext}."""
        artifact = plot_prediction_depth_distribution(
            depth_results, "sst2", output_dir=tmp_path
        )
        assert (
            artifact.figure_path
            == tmp_path / "sst2_prediction_depth_distribution.png"
        )
        assert (
            artifact.data_path == tmp_path / "sst2_prediction_depth_distribution.csv"
        )

    def test_figure_file_is_non_empty_png(
        self, depth_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """The saved figure should be a non-trivial PNG file."""
        artifact = plot_prediction_depth_distribution(
            depth_results, "sst2", output_dir=tmp_path
        )
        data = artifact.figure_path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(data) > 100

    def test_filters_to_requested_task(
        self, depth_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Rows belonging to other tasks should not appear in the
        companion CSV."""
        artifact = plot_prediction_depth_distribution(
            depth_results, "sst2", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            csv_rows = list(csv.DictReader(f))
        assert {row["task"] for row in csv_rows} == {"sst2"}

    def test_csv_row_count_matches_filtered_input(
        self, depth_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """The companion CSV should contain exactly the per-example
        rows that matched the requested task."""
        artifact = plot_prediction_depth_distribution(
            depth_results, "sst2", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            csv_rows = list(csv.DictReader(f))

        expected_count = sum(1 for r in depth_results if r["task"] == "sst2")
        assert len(csv_rows) == expected_count

    def test_csv_depths_match_input(
        self, depth_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """The multiset of prediction_depth values in the CSV should
        match the filtered input exactly."""
        artifact = plot_prediction_depth_distribution(
            depth_results, "sst2", output_dir=tmp_path
        )
        with open(artifact.data_path, newline="") as f:
            csv_rows = list(csv.DictReader(f))

        expected = sorted(
            int(r["prediction_depth"]) for r in depth_results if r["task"] == "sst2"
        )
        actual = sorted(int(row["prediction_depth"]) for row in csv_rows)
        assert actual == expected

    def test_creates_output_directory_if_missing(
        self, depth_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """output_dir need not exist beforehand."""
        output_dir = tmp_path / "nested" / "plots"
        assert not output_dir.exists()
        plot_prediction_depth_distribution(
            depth_results, "sst2", output_dir=output_dir
        )
        assert output_dir.exists()

    def test_raises_on_unknown_task(
        self, depth_results: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """A task_name with no matching rows should raise a clear
        error."""
        with pytest.raises(ValueError, match="No rows found for task"):
            plot_prediction_depth_distribution(
                depth_results, "unknown_task", output_dir=tmp_path
            )

    def test_raises_on_missing_required_keys(self, tmp_path: Path) -> None:
        """Rows missing pool_strategy or prediction_depth should raise
        a clear error rather than failing deep inside plotting code."""
        malformed = [{"task": "sst2", "pool_strategy": "last_token"}]
        with pytest.raises(ValueError, match="must contain"):
            plot_prediction_depth_distribution(malformed, "sst2", output_dir=tmp_path)
