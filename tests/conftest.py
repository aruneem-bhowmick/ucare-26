"""
Shared pytest fixtures for the test suite.

Provides mock model and tokenizer objects that mimic the Pythia-70m-deduped
architecture without requiring model downloads. All fixtures use
deterministic random seeds so that tensor values are reproducible across
test runs.

Architecture constants (from pythia-70m-deduped):
    - num_hidden_layers: 6
    - hidden_size: 512
    - num_attention_heads: 8
    - vocab_size: 50304
"""

import pytest
import torch
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Constants matching pythia-70m-deduped architecture
# ---------------------------------------------------------------------------

MOCK_NUM_LAYERS: int = 6
"""Number of transformer blocks in pythia-70m-deduped."""

MOCK_HIDDEN_SIZE: int = 512
"""Hidden representation dimensionality in pythia-70m-deduped."""

MOCK_VOCAB_SIZE: int = 50304
"""Vocabulary size of the GPT-NeoX tokenizer."""

MOCK_SEQ_LEN: int = 12
"""Typical short-sentence token count used in tests."""

MOCK_BATCH_SIZE: int = 1
"""Default batch size for single-input tests."""

FIXTURE_SEED: int = 123
"""Seed used within fixtures for deterministic tensor generation."""


# ---------------------------------------------------------------------------
# Hidden state fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hidden_states():
    """Create a mock hidden states tuple matching pythia-70m-deduped.

    Returns a tuple of 7 tensors (1 embedding output + 6 transformer
    block outputs), each of shape ``(1, 12, 512)``. Uses a fixed seed
    so that tensor values are deterministic across test runs.

    Returns:
        A tuple of 7 ``torch.Tensor`` objects.
    """
    gen = torch.Generator().manual_seed(FIXTURE_SEED)
    return tuple(
        torch.randn(
            MOCK_BATCH_SIZE, MOCK_SEQ_LEN, MOCK_HIDDEN_SIZE, generator=gen
        )
        for _ in range(MOCK_NUM_LAYERS + 1)
    )


@pytest.fixture
def mock_attention_mask():
    """Create a mock attention mask with left padding.

    Returns a tensor of shape ``(1, 12)`` where the first 2 positions
    are padding (0) and the remaining 10 positions are real tokens (1).
    This simulates a left-padded sequence.

    Returns:
        A ``torch.Tensor`` of shape ``(1, 12)``.
    """
    # 2 padding tokens on the left, 10 real tokens on the right
    mask = torch.zeros(MOCK_BATCH_SIZE, MOCK_SEQ_LEN, dtype=torch.long)
    mask[:, 2:] = 1
    return mask


@pytest.fixture
def mock_attention_mask_no_padding():
    """Create a mock attention mask with no padding (all ones).

    Returns a tensor of shape ``(1, 12)`` with all positions set to 1,
    indicating that every token in the sequence is a real token.

    Returns:
        A ``torch.Tensor`` of shape ``(1, 12)``.
    """
    return torch.ones(MOCK_BATCH_SIZE, MOCK_SEQ_LEN, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model and tokenizer mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model(mock_hidden_states):
    """Create a mock CausalLM model that returns plausible outputs.

    The mock model:
    - Has a ``config`` with ``num_hidden_layers=6`` and ``hidden_size=512``
    - Has parameters on CPU
    - When called, returns an object with a ``.hidden_states`` attribute
      containing the ``mock_hidden_states`` fixture

    Args:
        mock_hidden_states: Injected fixture providing synthetic hidden
            state tensors.

    Returns:
        A ``MagicMock`` instance configured to behave like a
        HuggingFace CausalLM.
    """
    model = MagicMock()

    # Configure the model's config object
    model.config.num_hidden_layers = MOCK_NUM_LAYERS
    model.config.hidden_size = MOCK_HIDDEN_SIZE

    # Make parameters() return a generator yielding a single CPU tensor
    # so that next(model.parameters()).device returns cpu
    param = torch.zeros(1)
    model.parameters.return_value = iter([param])

    # Configure the model's forward pass output
    output = MagicMock()
    output.hidden_states = mock_hidden_states
    model.return_value = output

    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with left-padding configured.

    The mock tokenizer:
    - Has ``eos_token`` set to ``"<|endoftext|>"``
    - Has ``pad_token`` initially set to ``None`` (as in GPT-NeoX)
    - Has ``padding_side`` initially set to ``"right"``
    - When called with text, returns a dict with ``input_ids`` and
      ``attention_mask`` tensors

    Returns:
        A ``MagicMock`` instance configured to behave like a
        HuggingFace tokenizer.
    """
    tokenizer = MagicMock()
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = None
    tokenizer.padding_side = "right"

    # When the tokenizer is called, return plausible tensors
    tokenizer.return_value = {
        "input_ids": torch.randint(0, MOCK_VOCAB_SIZE, (MOCK_BATCH_SIZE, MOCK_SEQ_LEN)),
        "attention_mask": torch.ones(MOCK_BATCH_SIZE, MOCK_SEQ_LEN, dtype=torch.long),
    }

    return tokenizer
