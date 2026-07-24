"""
End-to-end validation of the extract -> probe -> plot pipeline.

Runs the full chain against real Pythia-160M representations on two
task families — SST-2 (shallow-semantic classification) and a
LAMA/T-REx subset (factual-lookup classification) — the way
``src.extraction.smoke_test.run_pipeline_validation`` validated the
extraction cache against real checkpoints. Extraction is done manually
(hook capture, pooling, caching) rather than through
``ExtractionPipeline``, since the LAMA/T-REx subset requires custom
relation filtering that the generic extraction pipeline does not
support.

Usage:
    python -m src.probing.smoke_test
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset

from src.data.loaders import load_lama_trex, load_sst2
from src.data.tasks import TaskSpec, get_task_spec
from src.extraction.cache import load_representations, save_representations, verify_manifest
from src.extraction.hooks import HookManager
from src.extraction.pooling import pool_hidden_states
from src.extraction.smoke_test import set_seed
from src.models import get_model_spec, load_model
from src.probing.cross_validation import run_cross_validated_probe
from src.probing.pipeline import (
    ProbingPipeline,
    _group_by_pool_strategy,
    _select_rows,
    _stack_prediction_depths,
)
from src.probing.plotting import (
    PlotArtifact,
    plot_prediction_depth_distribution,
    plot_score_vs_layer,
)
from src.probing.probes import load_probing_config

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_KEY: str = "pythia-160m"
"""Model registry key validated by this smoke test."""

DEFAULT_SEED: int = 42
"""Default random seed for reproducibility."""

VALIDATION_TASKS: list[str] = ["sst2", "lama_trex"]
"""Task names validated end-to-end by this smoke test."""

VALIDATION_SPLIT: dict[str, str] = {"sst2": "validation", "lama_trex": "train"}
"""Dataset split to draw validation examples from, per task."""

VALIDATION_MAX_EXAMPLES: dict[str, int] = {"sst2": 64, "lama_trex": 90}
"""Number of examples to extract and probe per task."""

VALIDATION_BATCH_SIZE: int = 8
"""Batch size for the extraction forward passes."""

LAMA_TREX_POOL_SIZE: int = 3000
"""Size of the initial unfiltered LAMA/T-REx draw used to find frequent relations."""

LAMA_TREX_NUM_RELATIONS: int = 3
"""Number of most-frequent relations retained in the LAMA/T-REx validation subset."""

SELECTIVITY_THRESHOLD: float = 0.05
"""Minimum task-vs-control score gap required at some mid-depth layer."""

MID_LAYER_FRACTION: tuple[float, float] = (0.25, 0.75)
"""Fraction-of-depth range (inclusive) considered 'mid-layer' for the selectivity check."""

QUALITATIVE_EARLY_FRACTION: float = 0.25
"""Fraction of the shallowest layers averaged for the qualitative trend check."""

QUALITATIVE_LATE_FRACTION: float = 0.5
"""Fraction-of-depth cutoff above which layers are averaged for the qualitative trend check."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _select_lama_trex_subset(
    pool_size: int,
    num_relations: int,
    subset_size: int,
    seed: int,
) -> Dataset:
    """Build a small, cross-validation-friendly LAMA/T-REx subset.

    LAMA/T-REx spans dozens of relation types, so a uniform random
    sample of a few dozen examples would spread too thin across
    relations for stratified 5-fold cross-validation. Instead, this
    draws a larger unfiltered pool, keeps only the most frequent
    relations found in that pool, then subsamples down to
    *subset_size* so every retained class has enough examples per
    fold. No relation IDs are hardcoded, so this is robust to dataset
    revisions that add, remove, or rename relations.

    Args:
        pool_size: Number of examples to draw before relation
            filtering, used only to estimate relation frequency.
        num_relations: Number of most-frequent relations to keep.
        subset_size: Final number of examples to return. If the
            filtered pool is smaller than this, the whole filtered
            pool is returned instead.
        seed: Random seed for reproducible sampling.

    Returns:
        A ``Dataset`` restricted to the *num_relations* most frequent
        relations in the pool, subsampled to at most *subset_size*
        examples.
    """
    pool = load_lama_trex(split="train", max_examples=pool_size, seed=seed)
    counts = Counter(pool["predicate_id"])
    top_relations = {relation for relation, _ in counts.most_common(num_relations)}

    filtered = pool.filter(lambda ex: ex["predicate_id"] in top_relations)
    if subset_size < len(filtered):
        filtered = filtered.shuffle(seed=seed).select(range(subset_size))

    logger.info(
        "LAMA/T-REx validation subset: %d examples across %d relations (%s)",
        len(filtered),
        len(top_relations),
        sorted(top_relations),
    )
    return filtered


def _load_validation_dataset(task_name: str, max_examples: int, seed: int) -> Dataset:
    """Load the validation dataset for one task, dispatching by task name.

    Args:
        task_name: Short task identifier (``"sst2"`` or
            ``"lama_trex"``).
        max_examples: Number of examples to load.
        seed: Random seed for reproducible sampling.

    Returns:
        A ``Dataset`` with ``text``, ``label``, and ``example_id``
        columns.

    Raises:
        ValueError: If *task_name* has no registered validation
            dataset loader.
    """
    if task_name == "sst2":
        return load_sst2(
            split=VALIDATION_SPLIT["sst2"], max_examples=max_examples, seed=seed
        )
    if task_name == "lama_trex":
        return _select_lama_trex_subset(
            pool_size=LAMA_TREX_POOL_SIZE,
            num_relations=LAMA_TREX_NUM_RELATIONS,
            subset_size=max_examples,
            seed=seed,
        )
    raise ValueError(f"No validation dataset loader registered for task {task_name!r}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_task_representations(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    model_key: str,
    task_name: str,
    pool_strategy: str,
    split: str,
    output_dir: str | Path,
    seed: int,
    model_revision: str,
    batch_size: int = VALIDATION_BATCH_SIZE,
    max_seq_length: int = 512,
) -> Path:
    """Extract and cache pooled hidden-state representations for one task.

    Manually batches the dataset through a hook-based forward pass
    (mirroring the extraction pipeline validation in
    ``src.extraction.smoke_test``) rather than using
    ``ExtractionPipeline``, since the caller may have already filtered
    or subsampled *dataset* in a way the generic pipeline's loaders
    cannot reproduce. Only a single pooling strategy is exercised,
    since every task validated here declares one strategy in the task
    registry.

    Args:
        model: A loaded, eval-mode causal LM.
        tokenizer: The corresponding tokenizer.
        dataset: Dataset with ``text``, ``label``, ``example_id``
            columns.
        model_key: Short model identifier, used for the cache path.
        task_name: Short task identifier, used for the cache path.
        pool_strategy: Pooling strategy to apply (``"last_token"`` or
            ``"mean"``).
        split: Dataset split label recorded in the cache manifest.
        output_dir: Root directory for cached artifacts.
        seed: Random seed recorded in the cache manifest.
        model_revision: Model revision string recorded in the cache
            manifest.
        batch_size: Number of examples per forward pass.
        max_seq_length: Maximum token length before truncation.

    Returns:
        Path to the cache directory containing the saved
        safetensors files and manifest.
    """
    device = next(model.parameters()).device
    texts = list(dataset["text"])
    labels = list(dataset["label"])
    example_ids = list(dataset["example_id"])
    num_examples = len(texts)

    all_pooled: dict[int, list[torch.Tensor]] = {}
    all_token_counts: list[int] = []

    for batch_start in range(0, num_examples, batch_size):
        batch_end = min(batch_start + batch_size, num_examples)
        batch_texts = texts[batch_start:batch_end]

        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        attention_mask = encoded["attention_mask"]
        all_token_counts.extend(int(c) for c in attention_mask.sum(dim=1).tolist())

        with HookManager(model) as hm:
            with torch.no_grad():
                model(**encoded)
            hidden_states = hm.get_hidden_states()

        pooled = pool_hidden_states(hidden_states, attention_mask, strategy=pool_strategy)
        for layer_idx, tensor in pooled.items():
            all_pooled.setdefault(layer_idx, []).append(tensor.cpu())

    representations = {
        layer_idx: torch.cat(tensors, dim=0) for layer_idx, tensors in all_pooled.items()
    }

    cache_dir = save_representations(
        representations=representations,
        labels=labels,
        example_ids=example_ids,
        token_counts=all_token_counts,
        output_dir=output_dir,
        model_key=model_key,
        dataset_name=task_name,
        split=split,
        pool_strategy=pool_strategy,
        seed=seed,
        model_revision=model_revision,
    )
    logger.info(
        "Extracted and cached %d examples for task=%s to %s",
        num_examples,
        task_name,
        cache_dir,
    )
    return cache_dir


# ---------------------------------------------------------------------------
# Per-example prediction depth (for the histogram plot)
# ---------------------------------------------------------------------------


def _compute_example_depth_rows(
    cache_dir: str | Path,
    task_spec: TaskSpec,
    config: dict[str, Any],
    task_name: str,
) -> list[dict[str, Any]]:
    """Recompute per-example prediction-depth rows from a cached extraction.

    ``ProbingPipeline.run()`` only returns the per-layer summary table
    (with prediction depth aggregated to a per-pool-strategy mean),
    since that is the tidy artifact the pipeline is responsible for.
    The depth *distribution* plot needs one row per example instead,
    so this recomputes it directly from the cache using the same
    cross-validation and prediction-depth building blocks the pipeline
    itself uses.

    Args:
        cache_dir: Path to a cache directory produced by
            ``save_representations``.
        task_spec: Task specification determining classification vs.
            regression dispatch.
        config: Parsed probing configuration, as returned by
            ``load_probing_config``.
        task_name: Task label to attach to every row.

    Returns:
        A list of per-example rows with ``task``, ``pool_strategy``,
        ``example_id``, and ``prediction_depth`` keys, suitable for
        ``plot_prediction_depth_distribution``.
    """
    representations, manifest = load_representations(cache_dir)
    layer_indices = sorted(representations.keys())
    pool_groups = _group_by_pool_strategy(manifest)

    rows: list[dict[str, Any]] = []
    for pool_strategy, indices in pool_groups.items():
        y = np.asarray([manifest[i]["label"] for i in indices])

        layer_predictions: dict[int, np.ndarray] = {}
        for layer_idx in layer_indices:
            X = _select_rows(representations[layer_idx], indices)
            cv_result = run_cross_validated_probe(X, y, task_spec, config)
            layer_predictions[layer_idx] = cv_result.probe.predict(X)

        depths = _stack_prediction_depths(layer_predictions, layer_indices)

        for position, manifest_idx in enumerate(indices):
            rows.append(
                {
                    "task": task_name,
                    "pool_strategy": pool_strategy,
                    "example_id": manifest[manifest_idx]["example_id"],
                    "prediction_depth": int(depths[position]),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def _check_cache_alignment(cache_dir: str | Path, dataset: Dataset) -> None:
    """Verify cached representations load correctly and align with task labels.

    Confirms the cache is internally consistent (``verify_manifest``)
    and that its manifest reproduces the source dataset's labels
    exactly, in order — i.e. the extract/cache round trip did not
    drop, reorder, or corrupt any example.

    Args:
        cache_dir: Path to the cache directory to verify.
        dataset: The dataset the cache was built from.

    Raises:
        AssertionError: If the manifest length, label values, or
            per-layer tensor row counts do not match *dataset*.
    """
    verify_manifest(cache_dir)
    representations, manifest = load_representations(cache_dir)

    assert len(manifest) == len(dataset), (
        f"Manifest has {len(manifest)} entries, expected {len(dataset)}"
    )

    expected_labels = list(dataset["label"])
    actual_labels = [entry["label"] for entry in manifest]
    assert actual_labels == expected_labels, (
        "Cached manifest labels do not match the source dataset's labels"
    )

    for layer_idx, tensor in representations.items():
        assert tensor.shape[0] == len(dataset), (
            f"Layer {layer_idx} has {tensor.shape[0]} rows, expected {len(dataset)}"
        )

    logger.info(
        "Check 1 PASSED: cache is aligned with %d source examples", len(dataset)
    )


def _check_layerwise_training(rows: list[dict[str, Any]], expected_layers: set[int]) -> None:
    """Verify per-layer probes trained without error at every expected layer.

    Args:
        rows: Tidy per-layer result rows, as returned by
            ``ProbingPipeline.run()``.
        expected_layers: Layer indices that should each have produced
            exactly one row.

    Raises:
        AssertionError: If any expected layer is missing or any row's
            score is non-finite (e.g. ``NaN`` from a failed fit).
    """
    assert rows, "Probing pipeline produced no rows"

    covered_layers = {row["layer"] for row in rows}
    assert covered_layers == expected_layers, (
        f"Probed layers {sorted(covered_layers)} do not match "
        f"expected layers {sorted(expected_layers)}"
    )

    for row in rows:
        assert np.isfinite(row["score"]), (
            f"Non-finite score at layer {row['layer']}"
        )

    logger.info("Check 2 PASSED: all %d layers trained without error", len(expected_layers))


def _check_selectivity_above_threshold(
    rows: list[dict[str, Any]],
    threshold: float = SELECTIVITY_THRESHOLD,
    mid_layer_fraction: tuple[float, float] = MID_LAYER_FRACTION,
) -> None:
    """Verify selectivity is materially above zero for at least one mid-layer.

    A probe reading genuine task-relevant structure should
    substantially outperform its shuffled-label control (Hewitt &
    Liang, 2019); this confirms at least one layer in the
    *mid_layer_fraction* depth range clears that bar, so linear-
    decodability results cannot be attributed to probe memorization.

    Args:
        rows: Tidy per-layer result rows with ``layer`` and
            ``selectivity`` keys.
        threshold: Minimum required selectivity.
        mid_layer_fraction: Inclusive ``(low, high)`` fraction-of-depth
            range considered mid-layer.

    Raises:
        AssertionError: If no mid-layer row's selectivity exceeds
            *threshold*.
    """
    layers = sorted({row["layer"] for row in rows})
    depth_span = layers[-1] - layers[0]
    low = layers[0] + mid_layer_fraction[0] * depth_span
    high = layers[0] + mid_layer_fraction[1] * depth_span

    mid_rows = [row for row in rows if low <= row["layer"] <= high]
    best_selectivity = max(row["selectivity"] for row in mid_rows)

    assert best_selectivity > threshold, (
        f"Best mid-layer selectivity {best_selectivity:.4f} did not exceed "
        f"threshold {threshold} across layers {[r['layer'] for r in mid_rows]}"
    )

    logger.info(
        "Check 3 PASSED: max mid-layer selectivity=%.4f (threshold=%.4f)",
        best_selectivity,
        threshold,
    )


def _check_metrics_populated(rows: list[dict[str, Any]], task_spec: TaskSpec) -> None:
    """Verify margin, ECE, and prediction depth are populated for every layer.

    Args:
        rows: Tidy per-layer result rows with ``margin_mean``,
            ``ece``, and ``prediction_depth_mean`` keys.
        task_spec: Task specification determining whether margin/ECE
            are expected to be finite (classification) or ``NaN``
            (regression).

    Raises:
        AssertionError: If any expected metric is non-finite (or, for
            classification tasks, if margin/ECE are unexpectedly
            ``NaN``).
    """
    is_classification = task_spec.label_type == "classification"

    for row in rows:
        assert np.isfinite(row["prediction_depth_mean"]), (
            f"Non-finite prediction_depth_mean at layer {row['layer']}"
        )
        if is_classification:
            assert np.isfinite(row["margin_mean"]), (
                f"Non-finite margin_mean at layer {row['layer']}"
            )
            assert np.isfinite(row["ece"]), f"Non-finite ece at layer {row['layer']}"

    logger.info("Check 4 PASSED: margin/ECE/prediction-depth populated for every layer")


def _check_plots_written(score_artifact: PlotArtifact, depth_artifact: PlotArtifact) -> None:
    """Verify both plots were written to disk as non-empty PNGs with companion data.

    Args:
        score_artifact: Artifact returned by ``plot_score_vs_layer``.
        depth_artifact: Artifact returned by
            ``plot_prediction_depth_distribution``.

    Raises:
        AssertionError: If either figure is missing, not a valid PNG,
            trivially small, or missing its companion CSV.
    """
    for artifact in (score_artifact, depth_artifact):
        assert artifact.figure_path.exists(), f"Missing figure: {artifact.figure_path}"
        png_bytes = artifact.figure_path.read_bytes()
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", (
            f"{artifact.figure_path} is not a valid PNG"
        )
        assert len(png_bytes) > 100, f"{artifact.figure_path} is suspiciously small"

        assert artifact.data_path.exists(), f"Missing companion CSV: {artifact.data_path}"
        assert artifact.data_path.stat().st_size > 0, f"{artifact.data_path} is empty"

    logger.info(
        "Check 5 PASSED: plots written to %s and %s",
        score_artifact.figure_path,
        depth_artifact.figure_path,
    )


def _check_qualitative_trend(
    rows: list[dict[str, Any]],
    early_fraction: float = QUALITATIVE_EARLY_FRACTION,
    late_fraction: float = QUALITATIVE_LATE_FRACTION,
) -> dict[str, Any]:
    """Softly check probe accuracy rises from early layers into mid-depth.

    Literature expectations (Fartale et al., 2025; Meng et al., 2022)
    suggest both recall and shallow-semantic content become linearly
    accessible before the final layers, so probe accuracy should rise
    from the shallowest layers into mid-depth. This is a sanity check
    against a small validation subset, not a hard correctness gate: a
    mismatch is logged as a warning and returned as data rather than
    raised, since sample-size noise can break a real-but-small effect.

    Args:
        rows: Tidy per-layer result rows with ``layer`` and ``score``
            keys.
        early_fraction: Fraction of the shallowest layers averaged
            into the "early" comparison group.
        late_fraction: Fraction-of-depth cutoff; layers at or beyond
            this depth are averaged into the "late" comparison group.

    Returns:
        A dict with ``rises_by_mid_depth`` (bool), ``early_mean_score``,
        and ``late_mean_score``.
    """
    layers = sorted({row["layer"] for row in rows})
    scores_by_layer = {row["layer"]: row["score"] for row in rows}

    depth_span = layers[-1] - layers[0]
    early_cutoff = layers[0] + early_fraction * depth_span
    late_cutoff = layers[0] + late_fraction * depth_span

    early_scores = [scores_by_layer[l] for l in layers if l <= early_cutoff]
    late_scores = [scores_by_layer[l] for l in layers if l >= late_cutoff]

    early_mean = float(np.mean(early_scores))
    late_mean = float(np.mean(late_scores))
    rises = late_mean > early_mean

    if rises:
        logger.info(
            "Check 6 PASSED (qualitative): late-layer mean score %.4f > "
            "early-layer mean score %.4f",
            late_mean,
            early_mean,
        )
    else:
        logger.warning(
            "Check 6 (qualitative): expected rising-then-plateau pattern not "
            "observed on this validation subset (early=%.4f, late=%.4f)",
            early_mean,
            late_mean,
        )

    return {
        "rises_by_mid_depth": bool(rises),
        "early_mean_score": early_mean,
        "late_mean_score": late_mean,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_probing_pipeline_validation(
    model_key: str = DEFAULT_MODEL_KEY,
    tasks: list[str] | None = None,
    max_examples: dict[str, int] | None = None,
    extraction_output_dir: str | Path = "outputs/extractions",
    plots_output_dir: str | Path = "outputs/plots",
    probing_output_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """Validate the full extract -> probe -> plot pipeline against real checkpoints.

    For each task: extracts and caches hidden-state representations
    from a real Pythia checkpoint, runs the per-layer probing
    pipeline, writes both probing plots, and runs all six validation
    checks (cache alignment, layerwise training, selectivity,
    metric population, plot output, and a qualitative accuracy-trend
    sanity check).

    Args:
        model_key: Model registry key to validate.
        tasks: Task names to validate. Defaults to
            ``VALIDATION_TASKS`` (``sst2``, ``lama_trex``).
        max_examples: Per-task example counts. Defaults to
            ``VALIDATION_MAX_EXAMPLES``.
        extraction_output_dir: Root directory for cached extractions.
        plots_output_dir: Directory probing plots are written to.
        probing_output_dir: Directory probing result CSVs are written
            to. Defaults to the probing config's own ``output_dir``.
        seed: Random seed for reproducibility.

    Returns:
        A dict keyed by task name with per-task validation results:
        example count, the tidy per-layer rows, both plot paths, and
        the qualitative trend outcome.
    """
    if tasks is None:
        tasks = list(VALIDATION_TASKS)
    if max_examples is None:
        max_examples = dict(VALIDATION_MAX_EXAMPLES)

    set_seed(seed)

    probing_config = load_probing_config()
    if probing_output_dir is not None:
        probing_config = {**probing_config, "output_dir": str(probing_output_dir)}

    model_spec = get_model_spec(model_key)
    model, tokenizer = load_model(model_spec)

    results: dict[str, dict[str, Any]] = {}

    for task_name in tasks:
        logger.info("=" * 60)
        logger.info("Validating probing pipeline for task: %s", task_name)
        logger.info("=" * 60)

        task_spec = get_task_spec(task_name)
        split = VALIDATION_SPLIT.get(task_name, "validation")
        n_examples = max_examples.get(task_name, VALIDATION_MAX_EXAMPLES.get(task_name, 64))

        dataset = _load_validation_dataset(task_name, n_examples, seed)

        cache_dir = _extract_task_representations(
            model,
            tokenizer,
            dataset,
            model_key=model_key,
            task_name=task_name,
            pool_strategy=task_spec.extraction_position,
            split=split,
            output_dir=extraction_output_dir,
            seed=seed,
            model_revision=model_spec.revision,
        )
        _check_cache_alignment(cache_dir, dataset)

        pipeline = ProbingPipeline(model_key, task_name, probing_config)
        rows = pipeline.run(cache_dir)

        expected_layers = set(range(model_spec.num_layers + 1))
        _check_layerwise_training(rows, expected_layers)
        _check_selectivity_above_threshold(rows)
        _check_metrics_populated(rows, task_spec)

        example_depth_rows = _compute_example_depth_rows(
            cache_dir, task_spec, probing_config, task_name
        )
        score_artifact = plot_score_vs_layer(rows, task_name, output_dir=plots_output_dir)
        depth_artifact = plot_prediction_depth_distribution(
            example_depth_rows, task_name, output_dir=plots_output_dir
        )
        _check_plots_written(score_artifact, depth_artifact)

        trend = _check_qualitative_trend(rows)

        results[task_name] = {
            "num_examples": len(dataset),
            "num_layers": model_spec.num_layers + 1,
            "rows": rows,
            "score_plot": str(score_artifact.figure_path),
            "depth_plot": str(depth_artifact.figure_path),
            "qualitative_trend": trend,
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("ALL PROBING PIPELINE VALIDATION CHECKS PASSED")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_probing_pipeline_validation()
