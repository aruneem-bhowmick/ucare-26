"""
Plotting utilities for probing results.

Renders the two figures a layer-by-layer probing analysis needs directly
from tidy result tables: probe performance (or another per-layer metric)
as a function of layer depth, and the distribution of per-example
prediction depth. Every figure is saved alongside a CSV of exactly the
data that was plotted, so a figure can be regenerated from cached
numbers without rerunning any probes. Uses matplotlib's non-interactive
``Agg`` backend so plotting works in headless environments (compute
clusters, CI) with no display attached.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlotArtifact:
    """Paths to a saved figure and its underlying plotted data.

    Attributes:
        figure_path: Path to the saved PNG figure.
        data_path: Path to the CSV of exactly the data used to draw the
            figure, so the figure can be regenerated without recomputing
            or rerunning any probes.
    """

    figure_path: Path
    data_path: Path


def _filter_by_task(
    results: list[dict[str, Any]], task_name: str
) -> list[dict[str, Any]]:
    """Select rows belonging to a single task from a tidy result table.

    Args:
        results: Tidy result rows, each expected to carry a ``"task"``
            key.
        task_name: Task value to filter on.

    Returns:
        The subset of *results* whose ``"task"`` value equals
        *task_name*, in their original relative order.

    Raises:
        ValueError: If no rows match *task_name*.
    """
    rows = [row for row in results if row["task"] == task_name]
    if not rows:
        raise ValueError(f"No rows found for task {task_name!r} in results")
    return rows


def _group_by_pool_strategy(
    rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Group result rows by their ``pool_strategy`` value.

    Args:
        rows: Result rows, each expected to carry a ``"pool_strategy"``
            key.

    Returns:
        A dict mapping each distinct ``pool_strategy`` value to its
        list of rows, in their original relative order.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["pool_strategy"], []).append(row)
    return groups


def _save_figure_and_csv(
    fig: "plt.Figure",
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    output_dir: str | Path,
    stem: str,
) -> PlotArtifact:
    """Save a figure as PNG and its underlying rows as a companion CSV.

    Args:
        fig: The matplotlib figure to save.
        rows: The exact rows that were plotted, written out as a CSV so
            the figure is regenerable without recomputation.
        fieldnames: CSV column order.
        output_dir: Directory the figure and CSV are written into.
            Created if it does not already exist.
        stem: Shared filename stem for both the ``.png`` and ``.csv``
            outputs.

    Returns:
        A ``PlotArtifact`` with the paths to both saved files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure_path = output_path / f"{stem}.png"
    data_path = output_path / f"{stem}.csv"

    fig.savefig(figure_path)
    plt.close(fig)

    with open(data_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return PlotArtifact(figure_path=figure_path, data_path=data_path)


def plot_score_vs_layer(
    results: list[dict[str, Any]],
    task_name: str,
    metric: str = "score",
    output_dir: str | Path = "outputs/plots",
) -> PlotArtifact:
    """Plot a per-layer metric against layer depth for a single task.

    Filters *results* (a tidy table shaped like
    ``ProbingPipeline.run()``'s output) down to *task_name*, then draws
    one line per distinct ``pool_strategy`` present, each ordered by
    layer. When a ``f"{metric}_std"`` column is present on every
    filtered row (as ``score_std`` is for ``metric="score"``), a shaded
    ``±std`` band is drawn around that line; metrics without a
    companion ``_std`` column are plotted as a line only.

    Args:
        results: Tidy result rows with at least ``task``,
            ``pool_strategy``, ``layer``, and *metric* keys.
        task_name: Task to filter to and plot.
        metric: Column to plot against layer depth, e.g. ``"score"``,
            ``"selectivity"``, ``"margin_mean"``, ``"ece"``, or
            ``"prediction_depth_mean"``.
        output_dir: Directory the figure and companion CSV are written
            into. Created if it does not already exist.

    Returns:
        A ``PlotArtifact`` pointing at the saved
        ``{task_name}_{metric}_vs_layer.png`` figure and its companion
        CSV.

    Raises:
        ValueError: If no rows match *task_name*, or if *metric* is not
            a key present on the filtered rows.
    """
    rows = _filter_by_task(results, task_name)
    if metric not in rows[0]:
        raise ValueError(f"metric {metric!r} not present in results columns")

    std_key = f"{metric}_std"
    has_std = all(std_key in row for row in rows)

    groups = _group_by_pool_strategy(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_rows: list[dict[str, Any]] = []

    for pool_strategy in sorted(groups):
        group_rows = sorted(groups[pool_strategy], key=lambda row: row["layer"])
        layers = [row["layer"] for row in group_rows]
        values = [row[metric] for row in group_rows]

        ax.plot(layers, values, marker="o", label=pool_strategy)

        if has_std:
            stds = [row[std_key] for row in group_rows]
            lower = [v - s for v, s in zip(values, stds)]
            upper = [v + s for v, s in zip(values, stds)]
            ax.fill_between(layers, lower, upper, alpha=0.2)

        for row, value in zip(group_rows, values):
            plot_row = {
                "task": task_name,
                "pool_strategy": pool_strategy,
                "layer": row["layer"],
                metric: value,
            }
            if has_std:
                plot_row[std_key] = row[std_key]
            plot_rows.append(plot_row)

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs. layer — {task_name}")
    ax.legend(title="pool strategy")
    fig.tight_layout()

    fieldnames = ["task", "pool_strategy", "layer", metric]
    if has_std:
        fieldnames.append(std_key)

    artifact = _save_figure_and_csv(
        fig, plot_rows, fieldnames, output_dir, f"{task_name}_{metric}_vs_layer"
    )
    logger.info(
        "Wrote score-vs-layer plot for task=%s metric=%s to %s",
        task_name,
        metric,
        artifact.figure_path,
    )
    return artifact


def plot_prediction_depth_distribution(
    results: list[dict[str, Any]],
    task_name: str,
    output_dir: str | Path = "outputs/plots",
) -> PlotArtifact:
    """Plot a histogram of per-example prediction depth for a single task.

    Filters *results* (a per-example table, distinct from the per-layer
    summary table used by ``plot_score_vs_layer``) down to *task_name*,
    then draws one overlaid histogram per distinct ``pool_strategy``
    present, using integer-aligned bins since prediction depth is a
    discrete layer index.

    Args:
        results: Per-example rows with ``task``, ``pool_strategy``, and
            ``prediction_depth`` keys.
        task_name: Task to filter to and plot.
        output_dir: Directory the figure and companion CSV are written
            into. Created if it does not already exist.

    Returns:
        A ``PlotArtifact`` pointing at the saved
        ``{task_name}_prediction_depth_distribution.png`` figure and its
        companion CSV.

    Raises:
        ValueError: If no rows match *task_name*, or if the filtered
            rows are missing ``pool_strategy`` or ``prediction_depth``.
    """
    rows = _filter_by_task(results, task_name)
    if "pool_strategy" not in rows[0] or "prediction_depth" not in rows[0]:
        raise ValueError(
            "results rows must contain 'pool_strategy' and 'prediction_depth' "
            "keys for plot_prediction_depth_distribution"
        )

    groups = _group_by_pool_strategy(rows)
    all_depths = [int(row["prediction_depth"]) for row in rows]
    lo, hi = min(all_depths), max(all_depths)
    bins = [b - 0.5 for b in range(lo, hi + 2)]

    fig, ax = plt.subplots(figsize=(7, 5))
    for pool_strategy in sorted(groups):
        depths = [int(row["prediction_depth"]) for row in groups[pool_strategy]]
        ax.hist(depths, bins=bins, alpha=0.5, label=pool_strategy)

    ax.set_xlabel("Prediction depth (layer index)")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction depth distribution — {task_name}")
    ax.legend(title="pool strategy")
    fig.tight_layout()

    plot_rows = [
        {
            "task": task_name,
            "pool_strategy": row["pool_strategy"],
            "prediction_depth": row["prediction_depth"],
        }
        for row in rows
    ]
    fieldnames = ["task", "pool_strategy", "prediction_depth"]

    artifact = _save_figure_and_csv(
        fig,
        plot_rows,
        fieldnames,
        output_dir,
        f"{task_name}_prediction_depth_distribution",
    )
    logger.info(
        "Wrote prediction-depth distribution plot for task=%s to %s",
        task_name,
        artifact.figure_path,
    )
    return artifact
