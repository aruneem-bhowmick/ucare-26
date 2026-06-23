"""
Model registry and loading utilities for the Pythia model suite.

Provides a typed interface for looking up model architecture
specifications from the YAML registry and loading HuggingFace
models with consistent configuration (left-padding, eval mode,
precision control).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Path to the default model registry config relative to the project root.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a single Pythia model checkpoint.

    Attributes:
        key: Short identifier used in configs and output paths
            (e.g. ``"pythia-70m"``).
        hf_id: Full HuggingFace model identifier
            (e.g. ``"EleutherAI/pythia-70m-deduped"``).
        num_layers: Number of transformer blocks.
        hidden_size: Dimensionality of hidden representations.
        num_heads: Number of attention heads.
        intermediate_size: Dimensionality of the MLP intermediate
            layer.
        vocab_size: Size of the tokenizer vocabulary.
        max_positions: Maximum supported sequence length.
        architecture: Model class name
            (e.g. ``"GPTNeoXForCausalLM"``).
        positional_embedding: Type of positional embedding
            (e.g. ``"rotary"``).
        parallel_attn: Whether attention and MLP run in parallel.
        revision: Git revision / branch to load from HuggingFace.
        role: Intended deployment context
            (e.g. ``"debug"``, ``"primary"``, ``"hcc"``).
    """

    key: str
    hf_id: str
    num_layers: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    vocab_size: int
    max_positions: int
    architecture: str
    positional_embedding: str
    parallel_attn: bool
    revision: str
    role: str


def load_model_registry(
    config_path: str | Path | None = None,
) -> dict[str, ModelSpec]:
    """Load the full model registry from the YAML configuration file.

    Parses ``configs/models.yaml`` (or the path provided) and returns
    a dictionary mapping short model keys to their ``ModelSpec``
    dataclass instances. Common architecture fields defined under the
    ``common`` section are merged into each model entry.

    Args:
        config_path: Path to the YAML registry file. Defaults to
            ``configs/models.yaml`` relative to the project root.

    Returns:
        A dictionary mapping model keys (e.g. ``"pythia-70m"``) to
        their corresponding ``ModelSpec`` instances.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If required fields are missing from a model entry.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    common: dict[str, Any] = raw.get("common", {})
    models_raw: dict[str, dict[str, Any]] = raw["models"]

    registry: dict[str, ModelSpec] = {}
    for key, entry in models_raw.items():
        spec = ModelSpec(
            key=key,
            hf_id=entry["hf_id"],
            num_layers=entry["num_layers"],
            hidden_size=entry["hidden_size"],
            num_heads=entry["num_heads"],
            intermediate_size=entry["intermediate_size"],
            vocab_size=common.get("vocab_size", entry.get("vocab_size")),
            max_positions=common.get("max_positions", entry.get("max_positions")),
            architecture=common.get("architecture", entry.get("architecture")),
            positional_embedding=common.get(
                "positional_embedding", entry.get("positional_embedding")
            ),
            parallel_attn=common.get("parallel_attn", entry.get("parallel_attn")),
            revision=common.get("revision", entry.get("revision", "main")),
            role=entry.get("role", "primary"),
        )
        registry[key] = spec

    logger.info("Loaded model registry with %d entries", len(registry))
    return registry


def get_model_spec(
    key: str,
    config_path: str | Path | None = None,
) -> ModelSpec:
    """Look up a single model specification by its short key.

    Convenience wrapper around ``load_model_registry`` that returns
    the ``ModelSpec`` for the given key or raises a clear error if
    the key is not found.

    Args:
        key: Short model identifier (e.g. ``"pythia-70m"``).
        config_path: Path to the YAML registry file. Defaults to
            ``configs/models.yaml`` relative to the project root.

    Returns:
        The ``ModelSpec`` for the requested model.

    Raises:
        KeyError: If *key* is not present in the registry.
    """
    registry = load_model_registry(config_path)
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(
            f"Unknown model key {key!r}. Available models: {available}"
        )
    return registry[key]


def load_model(
    spec: ModelSpec,
    precision: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a Pythia model and tokenizer from a ``ModelSpec``.

    Downloads (or loads from cache) the HuggingFace model specified
    by *spec*, sets it to evaluation mode with the requested
    precision, and configures the tokenizer with left-padding using
    the EOS token as the pad token.

    Args:
        spec: A ``ModelSpec`` describing which model to load.
        precision: Torch dtype for model weights. Defaults to
            ``torch.float16``.

    Returns:
        A ``(model, tokenizer)`` tuple where the model is in eval
        mode and the tokenizer is configured for left-padding.
    """
    logger.info(
        "Loading model %s (revision=%s, precision=%s)",
        spec.hf_id,
        spec.revision,
        precision,
    )

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        torch_dtype=precision,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, revision=spec.revision
    )
    # GPT-NeoX models lack a dedicated pad token; reuse EOS.
    tokenizer.pad_token = tokenizer.eos_token
    # Left-padding keeps the final real token at index -1, which
    # simplifies downstream representation extraction.
    tokenizer.padding_side = "left"

    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer
