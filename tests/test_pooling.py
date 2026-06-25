"""
Tests for standalone pooling strategies.

Validates that ``last_token_pool``, ``mean_pool``, and
``pool_hidden_states`` produce correct shapes, handle padding
correctly, and dispatch to the right strategy.
"""

import pytest
import torch

from src.extraction.pooling import last_token_pool, mean_pool, pool_hidden_states


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 16
SEQ_LEN = 8


# ---------------------------------------------------------------------------
# TestLastTokenPool
# ---------------------------------------------------------------------------


class TestLastTokenPool:
    """Tests for last_token_pool."""

    def test_output_shape(self):
        """Output is (batch, hidden)."""
        hidden = torch.randn(3, SEQ_LEN, HIDDEN_SIZE)
        mask = torch.ones(3, SEQ_LEN, dtype=torch.long)
        result = last_token_pool(hidden, mask)
        assert result.shape == (3, HIDDEN_SIZE)

    def test_correct_token_with_left_padding(self):
        """With left padding, selects the rightmost real token."""
        hidden = torch.zeros(1, SEQ_LEN, HIDDEN_SIZE)
        # Place a known value at position 5 (last real token)
        hidden[0, 5, :] = 1.0

        mask = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        mask[0, 2:6] = 1  # tokens at positions 2-5 are real

        result = last_token_pool(hidden, mask)
        assert torch.allclose(result, torch.ones(1, HIDDEN_SIZE))

    def test_correct_token_with_no_padding(self):
        """With no padding, selects the last position."""
        hidden = torch.zeros(1, SEQ_LEN, HIDDEN_SIZE)
        hidden[0, -1, :] = 42.0

        mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

        result = last_token_pool(hidden, mask)
        assert torch.allclose(result, torch.full((1, HIDDEN_SIZE), 42.0))

    def test_batch_with_different_padding(self):
        """Batch of two sequences with different padding amounts."""
        hidden = torch.zeros(2, SEQ_LEN, HIDDEN_SIZE)
        # Sequence 0: last real token at position 5
        hidden[0, 5, :] = 1.0
        # Sequence 1: last real token at position 7 (no padding)
        hidden[1, 7, :] = 2.0

        mask = torch.zeros(2, SEQ_LEN, dtype=torch.long)
        mask[0, 2:6] = 1  # seq 0: positions 2-5
        mask[1, :] = 1     # seq 1: all positions

        result = last_token_pool(hidden, mask)
        assert result.shape == (2, HIDDEN_SIZE)
        assert torch.allclose(result[0], torch.ones(HIDDEN_SIZE))
        assert torch.allclose(result[1], torch.full((HIDDEN_SIZE,), 2.0))


# ---------------------------------------------------------------------------
# TestMeanPool
# ---------------------------------------------------------------------------


class TestMeanPool:
    """Tests for mean_pool."""

    def test_output_shape(self):
        """Output is (batch, hidden)."""
        hidden = torch.randn(3, SEQ_LEN, HIDDEN_SIZE)
        mask = torch.ones(3, SEQ_LEN, dtype=torch.long)
        result = mean_pool(hidden, mask)
        assert result.shape == (3, HIDDEN_SIZE)

    def test_mean_excludes_padding(self):
        """Padding positions should not contribute to the mean."""
        hidden = torch.ones(1, 4, HIDDEN_SIZE)
        # Set padding positions to large values that would skew the mean
        hidden[0, 0, :] = 100.0  # padding
        hidden[0, 1, :] = 100.0  # padding

        mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)  # first 2 are padding

        result = mean_pool(hidden, mask)
        # Mean of positions 2 and 3 (both value 1.0) = 1.0
        assert torch.allclose(result, torch.ones(1, HIDDEN_SIZE))

    def test_no_padding_equals_full_mean(self):
        """Without padding, mean_pool equals torch.mean over seq dim."""
        gen = torch.Generator().manual_seed(42)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, generator=gen)
        mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

        result = mean_pool(hidden, mask)
        expected = hidden.float().mean(dim=1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_single_non_padding_token(self):
        """With one real token, mean equals that token's representation."""
        hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE)
        # Only position 3 is a real token
        mask = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        mask[0, 3] = 1

        result = mean_pool(hidden, mask)
        expected = hidden[0, 3, :].float().unsqueeze(0)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# TestPoolHiddenStates
# ---------------------------------------------------------------------------


class TestPoolHiddenStates:
    """Tests for pool_hidden_states dispatcher."""

    @pytest.fixture
    def hidden_states_tuple(self):
        """Create a tuple of hidden state tensors (7 layers)."""
        gen = torch.Generator().manual_seed(99)
        return tuple(
            torch.randn(2, SEQ_LEN, HIDDEN_SIZE, generator=gen) for _ in range(7)
        )

    @pytest.fixture
    def attention_mask(self):
        """Attention mask with no padding."""
        return torch.ones(2, SEQ_LEN, dtype=torch.long)

    def test_returns_dict_keyed_by_layer_index(
        self, hidden_states_tuple, attention_mask
    ):
        """Result is a dict mapping layer indices to tensors."""
        result = pool_hidden_states(
            hidden_states_tuple, attention_mask, strategy="last_token"
        )
        assert isinstance(result, dict)
        assert all(isinstance(k, int) for k in result.keys())
        assert all(isinstance(v, torch.Tensor) for v in result.values())

    def test_layer_indices_none_returns_all(
        self, hidden_states_tuple, attention_mask
    ):
        """layer_indices=None returns all layers."""
        result = pool_hidden_states(
            hidden_states_tuple,
            attention_mask,
            strategy="last_token",
            layer_indices=None,
        )
        assert len(result) == 7
        assert set(result.keys()) == set(range(7))

    def test_layer_indices_subset(
        self, hidden_states_tuple, attention_mask
    ):
        """layer_indices=[0, 2] returns only those layers."""
        result = pool_hidden_states(
            hidden_states_tuple,
            attention_mask,
            strategy="mean",
            layer_indices=[0, 2],
        )
        assert set(result.keys()) == {0, 2}

    def test_invalid_strategy_raises_value_error(
        self, hidden_states_tuple, attention_mask
    ):
        """Unknown strategy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            pool_hidden_states(
                hidden_states_tuple,
                attention_mask,
                strategy="max_pool",
            )

    def test_output_tensor_shapes(
        self, hidden_states_tuple, attention_mask
    ):
        """Each pooled tensor has shape (batch, hidden)."""
        result = pool_hidden_states(
            hidden_states_tuple,
            attention_mask,
            strategy="last_token",
        )
        for tensor in result.values():
            assert tensor.shape == (2, HIDDEN_SIZE)

    def test_mean_strategy_dispatches_correctly(
        self, hidden_states_tuple, attention_mask
    ):
        """Mean strategy produces same result as calling mean_pool directly."""
        result = pool_hidden_states(
            hidden_states_tuple,
            attention_mask,
            strategy="mean",
            layer_indices=[0],
        )
        direct = mean_pool(hidden_states_tuple[0], attention_mask)
        assert torch.allclose(result[0], direct, atol=1e-6)

    def test_last_token_strategy_dispatches_correctly(
        self, hidden_states_tuple, attention_mask
    ):
        """Last-token strategy produces same result as calling last_token_pool directly."""
        result = pool_hidden_states(
            hidden_states_tuple,
            attention_mask,
            strategy="last_token",
            layer_indices=[3],
        )
        direct = last_token_pool(hidden_states_tuple[3], attention_mask)
        assert torch.allclose(result[3], direct, atol=1e-6)
