"""
FP16 caching with safetensors and JSONL manifest.

Provides functions for saving extracted representations to disk in
fp16 safetensors format with a JSONL manifest that records per-example
metadata. The cache layout is::

    {output_dir}/{model_key}/{dataset_name}/
      layer_00.safetensors
      layer_01.safetensors
      ...
      manifest.jsonl
"""

import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def save_representations(
    representations: dict[int, torch.Tensor],
    labels: list[int | float],
    example_ids: list[int],
    token_counts: list[int],
    output_dir: str | Path,
    model_key: str,
    dataset_name: str,
    split: str,
    pool_strategy: str,
    seed: int,
    model_revision: str = "main",
) -> Path:
    """Save extracted representations to disk as fp16 safetensors.

    Creates the directory ``{output_dir}/{model_key}/{dataset_name}/``
    and writes one safetensors file per layer plus a JSONL manifest
    with per-example metadata.

    Args:
        representations: Mapping from layer index to tensor of shape
            ``(num_examples, hidden)``.
        labels: Label for each example.
        example_ids: Unique identifier for each example.
        token_counts: Number of non-padding tokens per example.
        output_dir: Root output directory.
        model_key: Short model identifier for the subdirectory.
        dataset_name: Dataset name for the subdirectory.
        split: Dataset split (e.g. ``"validation"``).
        pool_strategy: Pooling strategy used (e.g. ``"last_token"``).
        seed: Random seed used for extraction.
        model_revision: Model revision string. Defaults to ``"main"``.

    Returns:
        Path to the cache directory containing the saved files.
    """
    cache_dir = Path(output_dir) / model_key / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    layer_indices = sorted(representations.keys())

    # Save each layer's tensor as fp16 safetensors
    for layer_idx in layer_indices:
        tensor = representations[layer_idx]
        filename = f"layer_{layer_idx:02d}.safetensors"
        filepath = cache_dir / filename
        save_file({"representations": tensor.half().contiguous()}, str(filepath))
        logger.debug("Saved %s (%s)", filepath, tuple(tensor.shape))

    # Write JSONL manifest with per-example metadata
    manifest_path = cache_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for i in range(len(example_ids)):
            entry = {
                "example_id": example_ids[i],
                "label": labels[i],
                "token_count": token_counts[i],
                "layer_indices": layer_indices,
                "pool_strategy": pool_strategy,
                "model_key": model_key,
                "model_revision": model_revision,
                "dataset_name": dataset_name,
                "split": split,
                "seed": seed,
                "torch_version": torch.__version__,
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(
        "Saved %d layers x %d examples to %s",
        len(layer_indices),
        len(example_ids),
        cache_dir,
    )
    return cache_dir


def load_representations(
    cache_dir: str | Path,
) -> tuple[dict[int, torch.Tensor], list[dict]]:
    """Load cached representations and manifest from disk.

    Discovers all ``layer_*.safetensors`` files in the cache
    directory and parses the ``manifest.jsonl`` file.

    Args:
        cache_dir: Path to the cache directory.

    Returns:
        A tuple of ``(representations, manifest_entries)`` where
        *representations* maps layer indices to tensors and
        *manifest_entries* is a list of dicts from the JSONL file.
    """
    cache_dir = Path(cache_dir)

    # Discover and load layer files
    representations: dict[int, torch.Tensor] = {}
    for safetensor_path in sorted(cache_dir.glob("layer_*.safetensors")):
        # Extract layer index from filename like "layer_03.safetensors"
        stem = safetensor_path.stem  # "layer_03"
        layer_idx = int(stem.split("_")[1])
        data = load_file(str(safetensor_path))
        representations[layer_idx] = data["representations"]

    # Parse manifest
    manifest_entries: list[dict] = []
    manifest_path = cache_dir / "manifest.jsonl"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    manifest_entries.append(json.loads(line))

    logger.info(
        "Loaded %d layers, %d manifest entries from %s",
        len(representations),
        len(manifest_entries),
        cache_dir,
    )
    return representations, manifest_entries


def verify_manifest(cache_dir: str | Path) -> bool:
    """Verify that a cache directory is internally consistent.

    Checks that:
    1. The manifest.jsonl file exists and is non-empty.
    2. All layer files referenced in the manifest exist.
    3. Tensor shapes are consistent with the manifest
       (number of examples matches manifest length, layer indices
       match referenced files).

    Args:
        cache_dir: Path to the cache directory to verify.

    Returns:
        ``True`` if the cache is valid.

    Raises:
        ValueError: If any consistency check fails.
    """
    cache_dir = Path(cache_dir)

    # Check manifest exists
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"Manifest file not found: {manifest_path}")

    # Parse manifest
    manifest_entries: list[dict] = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_entries.append(json.loads(line))

    if not manifest_entries:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    # Get expected layer indices from the first manifest entry
    expected_layers = manifest_entries[0]["layer_indices"]
    num_examples = len(manifest_entries)

    # Check all layer files exist and have correct shapes
    for layer_idx in expected_layers:
        layer_file = cache_dir / f"layer_{layer_idx:02d}.safetensors"
        if not layer_file.exists():
            raise ValueError(
                f"Layer file missing: {layer_file} "
                f"(referenced in manifest)"
            )

        data = load_file(str(layer_file))
        tensor = data["representations"]
        if tensor.shape[0] != num_examples:
            raise ValueError(
                f"Shape mismatch in {layer_file.name}: "
                f"expected {num_examples} examples, "
                f"got {tensor.shape[0]}"
            )

    # Check that no extra layer files exist beyond what's in the manifest
    actual_layer_files = sorted(cache_dir.glob("layer_*.safetensors"))
    actual_indices = []
    for p in actual_layer_files:
        actual_indices.append(int(p.stem.split("_")[1]))

    if set(actual_indices) != set(expected_layers):
        raise ValueError(
            f"Layer file mismatch: manifest references layers "
            f"{expected_layers}, but found files for layers "
            f"{actual_indices}"
        )

    logger.info("Cache verification passed: %s", cache_dir)
    return True
