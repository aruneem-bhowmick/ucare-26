"""
Smoke test for Pythia hidden state extraction.

This module verifies that the extraction pipeline works end-to-end by:
1. Loading a small Pythia model (pythia-70m-deduped) in float16
2. Tokenizing a dummy sentence with left-padding configuration
3. Running a forward pass with output_hidden_states=True
4. Validating the hidden state tuple structure
5. Extracting representations using two pooling strategies

The test is designed to run on resource-constrained hardware (e.g.,
Jetson Orin Nano) and serves as the initial milestone for the UCARE
2026-27 project on principled early-halting criteria in LLMs.

Usage:
    python -m src.extraction.smoke_test
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.loaders import load_sst2
from src.extraction.cache import load_representations, save_representations, verify_manifest
from src.extraction.hooks import HookManager
from src.extraction.pooling import pool_hidden_states
from src.models import get_model_spec, load_model

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str = "EleutherAI/pythia-70m-deduped"
"""HuggingFace identifier for the default Pythia model."""

DEFAULT_REVISION: str = "main"
"""Git revision / branch of the model to load."""

DEFAULT_PRECISION: torch.dtype = torch.float16
"""Default floating-point precision for model weights."""

DEFAULT_SEED: int = 42
"""Default random seed for reproducibility."""

DUMMY_SENTENCE: str = "The quick brown fox jumps over the lazy dog."
"""Short sentence used as the dummy input for the smoke test."""

VALIDATION_MODEL_KEYS: list[str] = ["pythia-70m", "pythia-160m"]
"""Model registry keys to validate in the pipeline validation."""

VALIDATION_MAX_EXAMPLES: int = 16
"""Number of SST-2 examples to use during pipeline validation."""

VALIDATION_POOL_STRATEGIES: list[str] = ["last_token", "mean"]
"""Pooling strategies to exercise during pipeline validation."""

VALIDATION_BATCH_SIZE: int = 8
"""Batch size for pipeline validation (16 examples = 2 batches)."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility across torch and CUDA.

    Configures both the CPU and (if available) all CUDA device random
    number generators to produce deterministic results.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Model and tokenizer loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str = DEFAULT_REVISION,
    precision: torch.dtype = DEFAULT_PRECISION,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a Pythia model and its tokenizer with proper configuration.

    Loads the model in the specified precision and sets it to evaluation
    mode. Configures the tokenizer with **left-padding** using the EOS
    token as the pad token, which is the standard practice for GPT-NeoX
    decoder-only models that lack a dedicated pad token.

    Args:
        model_name: HuggingFace model identifier.
            Defaults to ``"EleutherAI/pythia-70m-deduped"``.
        revision: Model revision or branch to load.
            Defaults to ``"main"``.
        precision: Torch dtype for model weights.
            Defaults to ``torch.float16``.

    Returns:
        A tuple of ``(model, tokenizer)`` where the model is in eval
        mode and the tokenizer is configured for left-padding.
    """
    logger.info(
        "Loading model %s (revision=%s, precision=%s)",
        model_name,
        revision,
        precision,
    )

    # Load model in the requested precision and set to eval mode
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=precision,
    )
    model.eval()

    # Load tokenizer and configure padding behaviour
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    # GPT-NeoX models do not define a pad token by default.
    # We reuse the EOS token for padding purposes.
    tokenizer.pad_token = tokenizer.eos_token

    # Left-padding ensures the final non-padding token is always at
    # index -1, which simplifies downstream representation extraction.
    tokenizer.padding_side = "left"

    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------


def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Run a forward pass and return all hidden states plus the attention mask.

    Tokenizes the input text, moves tensors to the model's device, and
    performs a forward pass with ``output_hidden_states=True`` inside a
    ``torch.no_grad()`` context to conserve memory.

    Args:
        model: A HuggingFace causal LM with ``output_hidden_states``
            support.
        tokenizer: The corresponding tokenizer, expected to be
            configured with left-padding.
        text: Input text string to process.

    Returns:
        A tuple of ``(hidden_states, attention_mask)`` where:

        - ``hidden_states`` is a tuple of tensors, one per layer
          (including the embedding output at index 0). For
          pythia-70m-deduped this is 7 tensors, each of shape
          ``(1, seq_len, 512)``.
        - ``attention_mask`` is a tensor of shape ``(1, seq_len)``
          indicating which positions are real tokens (1) vs. padding (0).
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Move all input tensors to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logger.info("Input tokenized: %d tokens", inputs["input_ids"].shape[1])

    # Forward pass with hidden state collection, no gradient computation
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states, inputs["attention_mask"]


# ---------------------------------------------------------------------------
# Hidden state validation
# ---------------------------------------------------------------------------


def validate_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    expected_num_layers: int,
    expected_hidden_size: int,
) -> None:
    """Validate the structure of extracted hidden states.

    Checks two invariants that must hold for any Pythia model:

    1. The hidden states tuple has length ``expected_num_layers + 1``
       (one embedding output plus one per transformer block).
    2. Every tensor in the tuple has the correct hidden dimension as
       its last axis.

    Args:
        hidden_states: Tuple of hidden state tensors from the forward
            pass.
        expected_num_layers: Number of transformer blocks in the model
            (e.g. 6 for pythia-70m-deduped).
        expected_hidden_size: Dimensionality of hidden representations
            (e.g. 512 for pythia-70m-deduped).

    Raises:
        AssertionError: If any structural check fails.
    """
    expected_count = expected_num_layers + 1
    actual_count = len(hidden_states)
    assert actual_count == expected_count, (
        f"Expected {expected_count} hidden state tensors "
        f"(embedding + {expected_num_layers} blocks), got {actual_count}"
    )

    for idx, tensor in enumerate(hidden_states):
        actual_dim = tensor.shape[-1]
        assert actual_dim == expected_hidden_size, (
            f"Hidden state at index {idx} has dimension {actual_dim}, "
            f"expected {expected_hidden_size}"
        )

    logger.info(
        "Hidden state validation passed: %d layers, hidden_size=%d",
        expected_num_layers,
        expected_hidden_size,
    )


# ---------------------------------------------------------------------------
# Representation pooling strategies
# ---------------------------------------------------------------------------


def get_last_non_padding_representation(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Extract the representation of the last non-padding token.

    For left-padded sequences, this finds the final real token position
    using the attention mask and returns its hidden state from the
    specified layer. With left-padding, the last non-padding token is
    always at the rightmost position in the sequence, but this function
    handles the general case for any padding configuration.

    Args:
        hidden_states: Tuple of hidden state tensors from the forward
            pass.
        attention_mask: Binary mask where 1 indicates a real token and
            0 indicates padding. Shape: ``(batch_size, seq_len)``.
        layer_idx: Which layer to extract from. ``-1`` means the final
            transformer block output. Defaults to ``-1``.

    Returns:
        Tensor of shape ``(batch_size, hidden_size)`` containing the
        last non-padding token's representation for each item in the
        batch.
    """
    # Select the hidden states for the requested layer
    layer_output = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_size)

    batch_size = layer_output.shape[0]

    # Find the index of the last non-padding token in each sequence.
    # attention_mask is 1 for real tokens and 0 for padding.
    # Flip the mask and find the first 1 from the right to locate the
    # last real token position.
    last_non_pad_idx = (
        attention_mask.shape[1]
        - 1
        - attention_mask.flip(dims=[1]).argmax(dim=1)
    )  # (batch_size,)

    # Gather the representation at the last non-padding position
    result = layer_output[torch.arange(batch_size, device=layer_output.device), last_non_pad_idx]

    return result  # (batch_size, hidden_size)


def get_mean_pooled_representation(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Compute mean-pooled representation over non-padding tokens.

    Averages the hidden states of all non-padding tokens (as indicated
    by the attention mask) from the specified layer. Padding positions
    are zeroed out before summation so they do not contribute to the
    mean.

    Args:
        hidden_states: Tuple of hidden state tensors from the forward
            pass.
        attention_mask: Binary mask where 1 indicates a real token and
            0 indicates padding. Shape: ``(batch_size, seq_len)``.
        layer_idx: Which layer to extract from. ``-1`` means the final
            transformer block output. Defaults to ``-1``.

    Returns:
        Tensor of shape ``(batch_size, hidden_size)`` containing the
        mean-pooled representation for each item in the batch.
    """
    # Select the hidden states for the requested layer
    layer_output = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_size)

    # Expand attention_mask to broadcast over the hidden dimension:
    # (batch_size, seq_len) -> (batch_size, seq_len, 1)
    mask_expanded = attention_mask.unsqueeze(-1).float()

    # Zero out padding positions and sum over the sequence dimension
    sum_repr = (layer_output.float() * mask_expanded).sum(dim=1)  # (batch_size, hidden_size)

    # Count non-padding tokens per sequence, with a small epsilon to
    # guard against division by zero (shouldn't happen in practice)
    token_count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch_size, 1)

    mean_repr = sum_repr / token_count  # (batch_size, hidden_size)

    return mean_repr


# ---------------------------------------------------------------------------
# Main smoke test orchestration
# ---------------------------------------------------------------------------


def run_smoke_test(
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str = DEFAULT_REVISION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Execute the full smoke test pipeline.

    This is the main orchestration function that:

    1. Sets random seeds for reproducibility.
    2. Loads the model and tokenizer.
    3. Extracts hidden states from a dummy sentence.
    4. Validates the hidden state structure against the model config.
    5. Extracts representations using both last-token and mean pooling.
    6. Prints a summary of results.

    Args:
        model_name: HuggingFace model identifier.
            Defaults to ``"EleutherAI/pythia-70m-deduped"``.
        revision: Model revision or branch.
            Defaults to ``"main"``.
        seed: Random seed for reproducibility.
            Defaults to 42.

    Returns:
        A dict with keys:

        - ``"hidden_states"``: the raw tuple of hidden state tensors
        - ``"attention_mask"``: the attention mask tensor
        - ``"last_token_repr"``: last non-padding token representation
        - ``"mean_pooled_repr"``: mean-pooled representation
        - ``"num_layers"``: number of hidden state tensors (including
          embedding)
        - ``"hidden_size"``: dimensionality of representations
    """
    # Step 1: Set seeds
    set_seed(seed)

    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        revision=revision,
    )

    # Step 3: Extract hidden states from a dummy input
    hidden_states, attention_mask = extract_hidden_states(
        model, tokenizer, DUMMY_SENTENCE
    )

    # Step 4: Validate structure against the model configuration
    num_hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    validate_hidden_states(hidden_states, num_hidden_layers, hidden_size)

    # Step 5: Extract pooled representations from the final layer
    last_token_repr = get_last_non_padding_representation(
        hidden_states, attention_mask
    )
    mean_pooled_repr = get_mean_pooled_representation(
        hidden_states, attention_mask
    )

    # Step 6: Print summary
    logger.info("=" * 60)
    logger.info("SMOKE TEST RESULTS")
    logger.info("=" * 60)
    logger.info("Model: %s", model_name)
    logger.info("Number of hidden state tensors: %d", len(hidden_states))
    logger.info(
        "  (1 embedding output + %d transformer blocks)",
        num_hidden_layers,
    )
    logger.info("Hidden size: %d", hidden_size)
    logger.info("Input sentence: %r", DUMMY_SENTENCE)
    logger.info(
        "Last non-padding token representation shape: %s",
        tuple(last_token_repr.shape),
    )
    logger.info(
        "Mean-pooled representation shape: %s",
        tuple(mean_pooled_repr.shape),
    )
    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED")
    logger.info("=" * 60)

    return {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "last_token_repr": last_token_repr,
        "mean_pooled_repr": mean_pooled_repr,
        "num_layers": len(hidden_states),
        "hidden_size": hidden_size,
    }


# ---------------------------------------------------------------------------
# Hook cross-check helper
# ---------------------------------------------------------------------------


def _cross_check_hooks_vs_output_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
) -> None:
    """Verify that HookManager captures the same hidden states as output_hidden_states.

    Runs two forward passes on the same input — one via
    ``output_hidden_states=True`` and one via ``HookManager`` — and
    asserts element-wise closeness.  This confirms hook registration
    is correct on the actual GPTNeoX model, not just mock modules.

    Args:
        model: A loaded GPTNeoX model in eval mode.
        tokenizer: The corresponding tokenizer.
        text: Input text to forward through the model.

    Raises:
        AssertionError: If the hidden states from the two methods diverge.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Pass A: built-in output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    reference = outputs.hidden_states

    # Pass B: HookManager
    with HookManager(model) as hm:
        with torch.no_grad():
            model(**inputs)
        hooked = hm.get_hidden_states()

    assert len(reference) == len(hooked), (
        f"Layer count mismatch: output_hidden_states returned {len(reference)} "
        f"tensors, HookManager returned {len(hooked)}"
    )

    for i in range(len(reference)):
        assert torch.allclose(reference[i], hooked[i], atol=1e-5), (
            f"Hidden states diverge at layer {i}: "
            f"max diff = {(reference[i] - hooked[i]).abs().max().item():.6e}"
        )

    logger.info(
        "Hook cross-check passed: %d layers match (atol=1e-5)",
        len(reference),
    )


# ---------------------------------------------------------------------------
# Pipeline validation
# ---------------------------------------------------------------------------


def run_pipeline_validation(
    model_keys: list[str] | None = None,
    task_name: str = "sst2",
    split: str = "validation",
    max_examples: int = VALIDATION_MAX_EXAMPLES,
    pool_strategies: list[str] | None = None,
    output_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """Validate the full extract -> save -> reload -> verify cycle on real models.

    Runs the hook-based extraction pipeline with real Pythia checkpoints
    and a small SST-2 subset, confirming hidden-state shapes, pooling
    output, fp16 caching, manifest metadata, and round-trip fidelity.

    Args:
        model_keys: Registry keys to validate. Defaults to
            ``VALIDATION_MODEL_KEYS`` (pythia-70m, pythia-160m).
        task_name: Dataset name for manifest metadata.
        split: Dataset split to load.
        max_examples: Number of examples to extract.
        pool_strategies: Pooling strategies to exercise. Defaults to
            ``VALIDATION_POOL_STRATEGIES`` (last_token, mean).
        output_dir: Root directory for cached artifacts. Defaults to
            a temporary directory.
        seed: Random seed for reproducibility.

    Returns:
        A dict keyed by ``"{model_key}/{pool_strategy}"`` with per-run
        metadata including shapes, dtypes, and cache paths.
    """
    if model_keys is None:
        model_keys = VALIDATION_MODEL_KEYS
    if pool_strategies is None:
        pool_strategies = VALIDATION_POOL_STRATEGIES

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ucare_validation_"))
    else:
        output_dir = Path(output_dir)
    logger.info("Validation output directory: %s", output_dir)

    set_seed(seed)

    # Load dataset once (shared across models)
    dataset = load_sst2(split=split, max_examples=max_examples, seed=seed)
    texts = dataset["text"]
    labels = dataset["label"]
    example_ids = dataset["example_id"]
    num_examples = len(texts)
    logger.info("Loaded %d examples from SST-2 (%s)", num_examples, split)

    results: dict[str, dict[str, Any]] = {}

    for model_key in model_keys:
        logger.info("=" * 60)
        logger.info("Validating model: %s", model_key)
        logger.info("=" * 60)

        # 1. Load model spec and model
        spec = get_model_spec(model_key)
        model, tokenizer = load_model(spec)
        device = next(model.parameters()).device

        # 2. Cross-check hooks vs output_hidden_states on first example
        _cross_check_hooks_vs_output_hidden_states(model, tokenizer, texts[0])

        for pool_strategy in pool_strategies:
            logger.info("--- Pool strategy: %s ---", pool_strategy)

            # Accumulate across batches
            all_pooled: dict[int, list[torch.Tensor]] = {}
            all_token_counts: list[int] = []

            for batch_start in range(0, num_examples, VALIDATION_BATCH_SIZE):
                batch_end = min(batch_start + VALIDATION_BATCH_SIZE, num_examples)
                batch_texts = texts[batch_start:batch_end]

                # Tokenize batch
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                attention_mask = encoded["attention_mask"]

                # Track token counts per example
                batch_token_counts = attention_mask.sum(dim=1).tolist()
                all_token_counts.extend([int(c) for c in batch_token_counts])

                # Hook-based extraction
                with HookManager(model) as hm:
                    with torch.no_grad():
                        model(**encoded)
                    hidden_states = hm.get_hidden_states()

                # Pool hidden states
                pooled = pool_hidden_states(
                    hidden_states, attention_mask, strategy=pool_strategy
                )

                # Accumulate per-layer tensors
                for layer_idx, tensor in pooled.items():
                    if layer_idx not in all_pooled:
                        all_pooled[layer_idx] = []
                    all_pooled[layer_idx].append(tensor.cpu())

            # Concatenate accumulated tensors
            representations: dict[int, torch.Tensor] = {}
            for layer_idx in sorted(all_pooled.keys()):
                representations[layer_idx] = torch.cat(all_pooled[layer_idx], dim=0)

            # ---------------------------------------------------------------
            # Check 1 — Hidden state shapes
            # ---------------------------------------------------------------
            expected_num_layers = spec.num_layers + 1  # embedding + transformer blocks
            assert len(representations) == expected_num_layers, (
                f"Expected {expected_num_layers} layers, got {len(representations)}"
            )
            for layer_idx, tensor in representations.items():
                assert tensor.shape == (num_examples, spec.hidden_size), (
                    f"Layer {layer_idx} shape {tuple(tensor.shape)} != "
                    f"expected ({num_examples}, {spec.hidden_size})"
                )
            logger.info(
                "Check 1 PASSED: %d layers, each (%d, %d)",
                len(representations),
                num_examples,
                spec.hidden_size,
            )

            # ---------------------------------------------------------------
            # Check 2 — Pooling shapes (confirmed by assertion above)
            # ---------------------------------------------------------------
            logger.info(
                "Check 2 PASSED: %s pooling produces (%d, %d) per layer",
                pool_strategy,
                num_examples,
                spec.hidden_size,
            )

            # ---------------------------------------------------------------
            # Check 3 — fp16 cache on disk
            # ---------------------------------------------------------------
            cache_dir = save_representations(
                representations=representations,
                labels=list(labels),
                example_ids=list(example_ids),
                token_counts=all_token_counts,
                output_dir=output_dir,
                model_key=model_key,
                dataset_name=f"{task_name}_{pool_strategy}",
                split=split,
                pool_strategy=pool_strategy,
                seed=seed,
                model_revision=spec.revision,
            )

            loaded_reprs, manifest_entries = load_representations(cache_dir)
            for layer_idx, tensor in loaded_reprs.items():
                assert tensor.dtype == torch.float16, (
                    f"Layer {layer_idx} dtype {tensor.dtype} != float16"
                )
            logger.info("Check 3 PASSED: cached files exist and dtype is float16")

            # ---------------------------------------------------------------
            # Check 4 — Manifest metadata
            # ---------------------------------------------------------------
            assert len(manifest_entries) == num_examples, (
                f"Manifest has {len(manifest_entries)} entries, "
                f"expected {num_examples}"
            )
            entry = manifest_entries[0]
            assert entry["model_key"] == model_key
            assert entry["dataset_name"] == f"{task_name}_{pool_strategy}"
            assert entry["pool_strategy"] == pool_strategy
            assert entry["split"] == split
            assert entry["seed"] == seed
            assert entry["layer_indices"] == sorted(representations.keys())
            logger.info("Check 4 PASSED: manifest metadata is correct")

            # Verify manifest consistency via verify_manifest
            verify_manifest(cache_dir)

            # ---------------------------------------------------------------
            # Check 5 — Round-trip fidelity
            # ---------------------------------------------------------------
            for layer_idx in representations:
                original_fp16 = representations[layer_idx].half()
                loaded_tensor = loaded_reprs[layer_idx]
                assert torch.equal(original_fp16, loaded_tensor), (
                    f"Round-trip mismatch at layer {layer_idx}: "
                    f"max diff = {(original_fp16 - loaded_tensor).abs().max().item():.6e}"
                )
            logger.info("Check 5 PASSED: round-trip fidelity confirmed")

            result_key = f"{model_key}/{pool_strategy}"
            results[result_key] = {
                "model_key": model_key,
                "pool_strategy": pool_strategy,
                "num_examples": num_examples,
                "num_layers": len(representations),
                "hidden_size": spec.hidden_size,
                "cache_dir": str(cache_dir),
                "dtype": "float16",
            }

        # Free GPU memory before loading next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Check 6 — Both models validated (confirmed by outer loop)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE VALIDATION SUMMARY")
    logger.info("=" * 60)
    for key, info in results.items():
        logger.info(
            "  %s: %d layers x %d examples, hidden=%d, dtype=%s",
            key,
            info["num_layers"],
            info["num_examples"],
            info["hidden_size"],
            info["dtype"],
        )
    logger.info("Check 6 PASSED: all %d model(s) validated", len(model_keys))
    logger.info("=" * 60)
    logger.info("ALL PIPELINE VALIDATION CHECKS PASSED")
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_smoke_test()
    run_pipeline_validation()
