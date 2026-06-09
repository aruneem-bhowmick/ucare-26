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
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    results = run_smoke_test()
