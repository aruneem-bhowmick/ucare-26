"""
Per-layer probing pipeline orchestrator.

Ties together the cross-validation harness, control-probe selectivity,
and probing metrics into a single per-(model, task) orchestrator that
consumes a cached extraction (see ``src.extraction.cache``) and
produces one tidy result row per ``(layer, pool_strategy)`` pair. This
is the primary artifact for comparing linear decodability, separability,
calibration, and prediction stability across layer depth and task
family — the pipeline never re-runs the model itself, it only reads
representations that were already extracted and cached.
"""

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.tasks import TaskSpec, get_task_spec
from src.extraction.cache import load_representations
from src.metrics.calibration import expected_calibration_error
from src.metrics.margins import classification_margin
from src.metrics.prediction_depth import prediction_depth
from src.probing.probes import load_probing_config
from src.probing.selectivity import compute_selectivity

logger = logging.getLogger(__name__)

# Columns written to the tidy result table, in column order.
RESULT_COLUMNS: list[str] = [
    "model",
    "task",
    "pool_strategy",
    "layer",
    "score",
    "score_std",
    "selectivity",
    "margin_mean",
    "ece",
    "prediction_depth_mean",
]


def _select_rows(tensor: torch.Tensor, indices: list[int]) -> np.ndarray:
    """Select a subset of rows from a cached representation tensor.

    Upcasts the result to float32 regardless of the tensor's stored
    precision, since cached representations are persisted as fp16 and
    sklearn estimators are not reliably fast or precise on float16
    input.

    Args:
        tensor: Representation tensor of shape ``(num_examples, hidden)``
            for a single layer, as returned by ``load_representations``.
        indices: Row indices to select, in the order they should appear
            in the output.

    Returns:
        A float32 numpy array of shape ``(len(indices), hidden)``.
    """
    return tensor[indices].detach().cpu().to(torch.float32).numpy()


def _group_by_pool_strategy(manifest: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Group manifest row indices by their recorded pooling strategy.

    A cache directory ordinarily contains a single pooling strategy
    (every entry written by one ``save_representations`` call shares
    the same ``pool_strategy`` value), but grouping generically means
    a cache directory that comes to hold more than one strategy is
    handled correctly without changes here.

    Args:
        manifest: List of per-example manifest entries, as returned by
            ``load_representations``. Row order must match the row
            order of every layer's representation tensor.

    Returns:
        A dict mapping each distinct ``pool_strategy`` value to the
        list of manifest row indices sharing that value, in original
        manifest order.
    """
    groups: dict[str, list[int]] = {}
    for idx, entry in enumerate(manifest):
        groups.setdefault(entry["pool_strategy"], []).append(idx)
    return groups


def _stack_prediction_depths(
    layer_predictions: dict[int, np.ndarray], layer_indices: list[int]
) -> np.ndarray:
    """Compute per-example prediction depth from stacked per-layer predictions.

    For each example, builds its sequence of top-1 predictions across
    layers (shallowest to deepest, per *layer_indices*) and applies
    ``prediction_depth`` to find the earliest layer at which the
    prediction settles into its final-layer value.

    Args:
        layer_predictions: Mapping from layer index to that layer's
            array of per-example top-1 predictions (classification) or
            predicted values (regression).
        layer_indices: Layer indices in shallow-to-deep order,
            determining the sequence order passed to
            ``prediction_depth``.

    Returns:
        An integer array of shape ``(num_examples,)`` with one
        prediction-depth value per example.
    """
    num_examples = len(layer_predictions[layer_indices[0]])
    depths = np.empty(num_examples, dtype=int)
    for example_idx in range(num_examples):
        sequence = [
            layer_predictions[layer_idx][example_idx] for layer_idx in layer_indices
        ]
        depths[example_idx] = prediction_depth(sequence)
    return depths


class ProbingPipeline:
    """Per-layer probing orchestrator for a single (model, task) pair.

    Loads a cached extraction, trains a cross-validated linear probe
    (with control-probe selectivity, margin, and calibration) at every
    ``(layer, pool_strategy)`` present in the cache, aggregates the
    results into a single tidy table, and writes that table to disk as
    CSV. The model itself is never loaded or re-run; this class strictly
    consumes representations that were already extracted and cached.

    Args:
        model_key: Short model identifier, used only as a label in the
            output table and output filename (the model is not loaded).
        task_name: Short task identifier, resolved against the task
            registry to determine classification-vs-regression
            dispatch and other probing behavior.
        config: Parsed probing configuration, as returned by
            ``load_probing_config``. Defaults to loading
            ``configs/probing.yaml`` when omitted.
    """

    def __init__(
        self,
        model_key: str,
        task_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model_key = model_key
        self.task_name = task_name
        self.task_spec: TaskSpec = get_task_spec(task_name)
        self.config: dict[str, Any] = (
            config if config is not None else load_probing_config()
        )

    def _probe_layer(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[dict[str, float], np.ndarray]:
        """Run selectivity, margin, and calibration for one layer's features.

        Trains the cross-validated task probe via ``compute_selectivity``
        (which internally performs the cross-validation run), then
        derives margin and calibration from the same fitted probe rather
        than retraining, so each layer is only cross-validated once.

        Args:
            X: Feature matrix of shape ``(num_examples, hidden)`` for a
                single layer and pool strategy.
            y: Target labels or values of shape ``(num_examples,)``.

        Returns:
            A ``(metrics, predictions)`` pair, where *metrics* holds
            ``score``, ``score_std``, ``selectivity``, ``margin_mean``,
            and ``ece``, and *predictions* is the fitted probe's
            per-example prediction array (used later for prediction
            depth). ``margin_mean`` and ``ece`` are ``NaN`` for
            regression tasks, since both are classification-only
            quantities.
        """
        selectivity_result = compute_selectivity(X, y, self.task_spec, self.config)
        cv_result = selectivity_result.task_result
        probe = cv_result.probe

        if self.task_spec.label_type == "classification":
            margin_mean = float(np.mean(classification_margin(probe, X, y)))
            ece = expected_calibration_error(probe, X, y)
        else:
            margin_mean = float("nan")
            ece = float("nan")

        metrics = {
            "score": cv_result.mean_score,
            "score_std": cv_result.std_score,
            "selectivity": selectivity_result.selectivity,
            "margin_mean": margin_mean,
            "ece": ece,
        }
        return metrics, probe.predict(X)

    def _write_csv(self, rows: list[dict[str, Any]]) -> Path:
        """Write the tidy result table to the configured output directory.

        Args:
            rows: Tidy result rows, one per ``(layer, pool_strategy)``.

        Returns:
            Path to the written CSV file,
            ``{output_dir}/{model_key}_{task_name}.csv``.
        """
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.model_key}_{self.task_name}.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Wrote probing results to %s", output_path)
        return output_path

    def run(self, cache_dir: str | Path) -> list[dict[str, Any]]:
        """Run the full per-layer probing pipeline against a cached extraction.

        Loads the cached representations and manifest, trains a
        cross-validated probe at every ``(layer, pool_strategy)``
        combination present, computes selectivity/margin/calibration
        for each, then computes per-example prediction depth from the
        stacked per-layer predictions once every layer has been probed.
        The resulting tidy table is written to disk and also returned.

        Args:
            cache_dir: Path to a cache directory produced by
                ``src.extraction.cache.save_representations``.

        Returns:
            The tidy result table as a list of row dicts, one per
            ``(layer, pool_strategy)`` combination, with columns
            matching ``RESULT_COLUMNS``.

        Raises:
            ValueError: If the cache directory contains no cached
                layers or no manifest entries.
        """
        representations, manifest = load_representations(cache_dir)
        if not representations:
            raise ValueError(f"No cached representations found in {cache_dir}")
        if not manifest:
            raise ValueError(f"No manifest entries found in {cache_dir}")

        layer_indices = sorted(representations.keys())
        pool_groups = _group_by_pool_strategy(manifest)

        rows: list[dict[str, Any]] = []
        for pool_strategy in sorted(pool_groups):
            indices = pool_groups[pool_strategy]
            y = np.asarray([manifest[i]["label"] for i in indices])

            layer_metrics: dict[int, dict[str, float]] = {}
            layer_predictions: dict[int, np.ndarray] = {}
            for layer_idx in layer_indices:
                X = _select_rows(representations[layer_idx], indices)
                metrics, predictions = self._probe_layer(X, y)
                layer_metrics[layer_idx] = metrics
                layer_predictions[layer_idx] = predictions

            depths = _stack_prediction_depths(layer_predictions, layer_indices)
            depth_mean = float(np.mean(depths))

            for layer_idx in layer_indices:
                rows.append(
                    {
                        "model": self.model_key,
                        "task": self.task_name,
                        "pool_strategy": pool_strategy,
                        "layer": layer_idx,
                        **layer_metrics[layer_idx],
                        "prediction_depth_mean": depth_mean,
                    }
                )

        logger.info(
            "Probing pipeline complete: model=%s, task=%s, %d rows",
            self.model_key,
            self.task_name,
            len(rows),
        )
        self._write_csv(rows)
        return rows
