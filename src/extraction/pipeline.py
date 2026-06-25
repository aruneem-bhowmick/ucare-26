"""
End-to-end extraction pipeline orchestrator.

Ties together model loading, data loading, hook-based hidden state
capture, pooling, and fp16 caching into a single ``ExtractionPipeline``
class that can be configured via ``configs/extraction.yaml``.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data import (
    generate_dyck,
    generate_modular_arithmetic,
    generate_periodic_table,
    get_task_spec,
    load_lama_trex,
    load_mrpc,
    load_sst2,
)
from src.data.tasks import TaskSpec
from src.extraction.cache import save_representations
from src.extraction.hooks import HookManager
from src.extraction.pooling import pool_hidden_states
from src.models import ModelSpec, get_model_spec, load_model

logger = logging.getLogger(__name__)

# Path to the default extraction config relative to the project root.
_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "extraction.yaml"
)

# Mapping from task name to its data-loading function.
_TASK_LOADERS: dict[str, Any] = {
    "sst2": load_sst2,
    "mrpc": load_mrpc,
    "lama_trex": load_lama_trex,
}

_TASK_GENERATORS: dict[str, Any] = {
    "dyck": generate_dyck,
    "modular_arithmetic": generate_modular_arithmetic,
    "periodic_table": generate_periodic_table,
}


def _load_extraction_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load extraction configuration from YAML.

    Args:
        config_path: Path to the extraction config file.
            Defaults to ``configs/extraction.yaml``.

    Returns:
        The ``extraction`` section of the config as a dict.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return raw.get("extraction", {})


class ExtractionPipeline:
    """Orchestrates hidden state extraction from a model on a task dataset.

    Loads the model, tokenizer, and dataset, then iterates through
    the data in batches, capturing hidden states via hooks, pooling
    them, and saving the results to disk.

    Args:
        model_key: Short model identifier (e.g. ``"pythia-70m"``).
        task_name: Short task identifier (e.g. ``"sst2"``).
        config_path: Path to ``extraction.yaml``. ``None`` uses the
            default location.
    """

    def __init__(
        self,
        model_key: str,
        task_name: str,
        config_path: str | Path | None = None,
    ) -> None:
        self.model_spec: ModelSpec = get_model_spec(model_key)
        self.task_spec: TaskSpec = get_task_spec(task_name)

        config = _load_extraction_config(config_path)
        self.batch_size: int = config.get("batch_size", 32)
        self.max_seq_length: int = config.get("max_seq_length", 512)
        self.pool_strategy: str = config.get("pooling", "last_token")
        self.output_dir: str = config.get("output_dir", "outputs/extractions")
        self.seed: int = config.get("seed", 42)

        # Parse layer selection: -1 means all layers, otherwise a list
        layers_cfg = config.get("layers", -1)
        if layers_cfg == -1:
            self.layer_indices: list[int] | None = None
        elif isinstance(layers_cfg, list):
            self.layer_indices = layers_cfg
        else:
            self.layer_indices = None

    def _load_dataset(
        self,
        split: str,
        max_examples: int | None,
    ) -> Any:
        """Load the dataset for the configured task.

        Args:
            split: Dataset split to load (e.g. ``"validation"``).
            max_examples: Maximum number of examples to load.

        Returns:
            A ``datasets.Dataset`` with at least ``text``, ``label``,
            and ``example_id`` columns.
        """
        task_name = self.task_spec.name

        if task_name in _TASK_LOADERS:
            return _TASK_LOADERS[task_name](
                split=split,
                max_examples=max_examples,
                seed=self.seed,
            )
        elif task_name in _TASK_GENERATORS:
            kwargs: dict[str, Any] = {
                "num_examples": max_examples or 1000,
                "seed": self.seed,
            }
            return _TASK_GENERATORS[task_name](**kwargs)
        else:
            raise ValueError(
                f"No loader or generator registered for task {task_name!r}"
            )

    def run(
        self,
        split: str = "validation",
        max_examples: int | None = None,
    ) -> Path:
        """Execute the full extraction pipeline.

        1. Load model and tokenizer.
        2. Load dataset.
        3. Iterate batches with hook-based hidden state capture.
        4. Pool hidden states per layer.
        5. Save to cache and return the cache directory path.

        Args:
            split: Dataset split to process.
            max_examples: Cap on the number of examples.

        Returns:
            Path to the cache directory containing safetensors
            and manifest.
        """
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Load model and tokenizer
        model, tokenizer = load_model(self.model_spec)
        device = next(model.parameters()).device

        # Load dataset
        dataset = self._load_dataset(split, max_examples)
        num_examples = len(dataset)
        logger.info("Loaded %d examples for task %s", num_examples, self.task_spec.name)

        # Accumulators: {layer_idx: list of (batch_hidden,)} tensors
        all_pooled: dict[int, list[torch.Tensor]] = {}
        all_labels: list[int | float] = []
        all_example_ids: list[int] = []
        all_token_counts: list[int] = []

        # Process in batches
        for batch_start in range(0, num_examples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_examples)
            batch_texts = [dataset[i]["text"] for i in range(batch_start, batch_end)]
            batch_labels = [dataset[i]["label"] for i in range(batch_start, batch_end)]
            batch_ids = [dataset[i]["example_id"] for i in range(batch_start, batch_end)]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            attention_mask = inputs["attention_mask"]

            # Compute token counts (number of non-padding tokens per example)
            token_counts = attention_mask.sum(dim=1).tolist()

            # Forward pass with hooks
            with HookManager(model, layer_indices=self.layer_indices) as hm:
                with torch.no_grad():
                    model(**inputs)
                hidden_states = hm.get_hidden_states()

            # Pool hidden states
            pooled = pool_hidden_states(
                hidden_states,
                attention_mask,
                strategy=self.pool_strategy,
                layer_indices=None,  # pool all captured layers
            )

            # Accumulate
            for layer_idx, tensor in pooled.items():
                if layer_idx not in all_pooled:
                    all_pooled[layer_idx] = []
                all_pooled[layer_idx].append(tensor.cpu())

            all_labels.extend(batch_labels)
            all_example_ids.extend(batch_ids)
            all_token_counts.extend([int(c) for c in token_counts])

        # Concatenate accumulated tensors
        representations: dict[int, torch.Tensor] = {}
        for layer_idx, tensors in all_pooled.items():
            representations[layer_idx] = torch.cat(tensors, dim=0)

        # Save to cache
        cache_dir = save_representations(
            representations=representations,
            labels=all_labels,
            example_ids=all_example_ids,
            token_counts=all_token_counts,
            output_dir=self.output_dir,
            model_key=self.model_spec.key,
            dataset_name=self.task_spec.name,
            split=split,
            pool_strategy=self.pool_strategy,
            seed=self.seed,
            model_revision=self.model_spec.revision,
        )

        logger.info("Extraction complete: %s", cache_dir)
        return cache_dir
