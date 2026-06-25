"""
Standalone pooling strategies for hidden state representations.

Provides last-token and mean-pooling functions that operate on
single-layer tensors, plus a dispatcher that applies a chosen
strategy across multiple layers. The core math is identical to
the inline implementations in ``smoke_test.py`` but refactored
to accept a single hidden-state tensor rather than the full tuple.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def last_token_pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract the representation of the last non-padding token.

    For left-padded sequences, this finds the final real token
    position using the attention mask and returns its hidden state.

    Args:
        hidden_state: Hidden state tensor of shape
            ``(batch, seq, hidden)`` from a single layer.
        attention_mask: Binary mask of shape ``(batch, seq)`` where
            1 indicates a real token and 0 indicates padding.

    Returns:
        Tensor of shape ``(batch, hidden)`` containing the last
        non-padding token's representation for each item in the
        batch.
    """
    batch_size = hidden_state.shape[0]

    # Find index of the last non-padding token in each sequence.
    # Flip the mask and find the first 1 from the right.
    last_non_pad_idx = (
        attention_mask.shape[1]
        - 1
        - attention_mask.flip(dims=[1]).argmax(dim=1)
    )  # (batch_size,)

    result = hidden_state[
        torch.arange(batch_size, device=hidden_state.device),
        last_non_pad_idx,
    ]

    return result  # (batch_size, hidden)


def mean_pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute mean-pooled representation over non-padding tokens.

    Averages the hidden states of all non-padding tokens (as
    indicated by the attention mask). Padding positions are zeroed
    out before summation so they do not contribute to the mean.

    Args:
        hidden_state: Hidden state tensor of shape
            ``(batch, seq, hidden)`` from a single layer.
        attention_mask: Binary mask of shape ``(batch, seq)`` where
            1 indicates a real token and 0 indicates padding.

    Returns:
        Tensor of shape ``(batch, hidden)`` containing the
        mean-pooled representation for each item in the batch.
    """
    # Expand mask for broadcasting: (batch, seq) -> (batch, seq, 1)
    mask_expanded = attention_mask.unsqueeze(-1).float()

    # Zero out padding positions and sum over the sequence dimension
    sum_repr = (hidden_state.float() * mask_expanded).sum(dim=1)  # (batch, hidden)

    # Count non-padding tokens per sequence, clamped to avoid division by zero
    token_count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)

    mean_repr = sum_repr / token_count  # (batch, hidden)

    return mean_repr


def pool_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    strategy: str,
    layer_indices: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Apply a pooling strategy across multiple layers.

    Dispatches to ``last_token_pool`` or ``mean_pool`` for each
    requested layer and returns a dictionary mapping layer indices
    to their pooled representations.

    Args:
        hidden_states: Tuple of hidden state tensors, one per layer
            (including embedding at index 0).
        attention_mask: Binary mask of shape ``(batch, seq)``.
        strategy: Pooling strategy name. Must be ``"last_token"``
            or ``"mean"``.
        layer_indices: Which layers to pool. ``None`` means all
            layers in the tuple.

    Returns:
        Dictionary mapping layer index to pooled tensor of shape
        ``(batch, hidden)``.

    Raises:
        ValueError: If *strategy* is not ``"last_token"`` or
            ``"mean"``.
    """
    if strategy == "last_token":
        pool_fn = last_token_pool
    elif strategy == "mean":
        pool_fn = mean_pool
    else:
        raise ValueError(
            f"Unknown pooling strategy {strategy!r}. "
            f"Must be 'last_token' or 'mean'."
        )

    if layer_indices is None:
        indices = list(range(len(hidden_states)))
    else:
        indices = layer_indices

    result: dict[int, torch.Tensor] = {}
    for idx in indices:
        result[idx] = pool_fn(hidden_states[idx], attention_mask)

    return result
