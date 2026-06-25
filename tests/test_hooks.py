"""
Tests for hook-based hidden state capture.

Validates that ``HookManager`` correctly registers and removes forward
hooks on mock GPTNeoX model components, captures tensors of the expected
shape, and handles layer subset selection and error conditions.
"""

import pytest
import torch
from torch import nn
from unittest.mock import MagicMock

from src.extraction.hooks import HookManager


# ---------------------------------------------------------------------------
# Helpers: minimal GPTNeoX-like model structure
# ---------------------------------------------------------------------------

MOCK_NUM_LAYERS = 6
MOCK_HIDDEN_SIZE = 32
MOCK_SEQ_LEN = 8
MOCK_BATCH_SIZE = 2


class FakeEmbedding(nn.Module):
    """Simple embedding layer that returns a fixed-shape tensor."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        # Need a real parameter so the module is properly initialised
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        return torch.randn(batch, seq, self._hidden_size)


class FakeGPTNeoXLayer(nn.Module):
    """Mimics GPTNeoXLayer: forward returns (residual_tensor, )."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple:
        # Return a tuple where element 0 is the residual stream
        output = torch.randn_like(hidden_states)
        return (output,)


class FakeGPTNeoXModel(nn.Module):
    """Minimal GPTNeoX backbone with embed_in and layers."""

    def __init__(self, num_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_in = FakeEmbedding(hidden_size)
        self.layers = nn.ModuleList(
            [FakeGPTNeoXLayer(hidden_size) for _ in range(num_layers)]
        )


class FakeGPTNeoXForCausalLM(nn.Module):
    """Minimal causal LM wrapper with a gpt_neox attribute."""

    def __init__(self, num_layers: int = MOCK_NUM_LAYERS, hidden_size: int = MOCK_HIDDEN_SIZE) -> None:
        super().__init__()
        self.gpt_neox = FakeGPTNeoXModel(num_layers, hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden = self.gpt_neox.embed_in(input_ids)
        for layer in self.gpt_neox.layers:
            hidden = layer(hidden)[0]
        return hidden


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_model():
    """Create a fake GPTNeoX causal LM."""
    model = FakeGPTNeoXForCausalLM()
    model.eval()
    return model


@pytest.fixture
def dummy_input():
    """Create a dummy input_ids tensor."""
    return torch.randint(0, 100, (MOCK_BATCH_SIZE, MOCK_SEQ_LEN))


# ---------------------------------------------------------------------------
# TestHookManager
# ---------------------------------------------------------------------------


class TestHookManager:
    """Tests for HookManager context manager."""

    def test_captures_correct_number_of_tensors_all_layers(
        self, fake_model, dummy_input
    ):
        """Hooks on all layers produce embedding + N layer tensors."""
        with HookManager(fake_model) as hm:
            fake_model(dummy_input)
            states = hm.get_hidden_states()

        # 1 embedding + 6 layers = 7 tensors
        assert len(states) == MOCK_NUM_LAYERS + 1

    def test_captured_tensor_shapes(self, fake_model, dummy_input):
        """Each captured tensor has shape (batch, seq, hidden)."""
        with HookManager(fake_model) as hm:
            fake_model(dummy_input)
            states = hm.get_hidden_states()

        for tensor in states:
            assert tensor.shape == (MOCK_BATCH_SIZE, MOCK_SEQ_LEN, MOCK_HIDDEN_SIZE)

    def test_layer_indices_subset(self, fake_model, dummy_input):
        """Selecting specific layers only captures embedding + those layers."""
        selected = [0, 2, 5]
        with HookManager(fake_model, layer_indices=selected) as hm:
            fake_model(dummy_input)
            states = hm.get_hidden_states()

        # 1 embedding + 3 selected layers = 4 tensors
        assert len(states) == 1 + len(selected)

    def test_clear_resets_captured_state(self, fake_model, dummy_input):
        """Calling clear() empties the captured tensors."""
        with HookManager(fake_model) as hm:
            fake_model(dummy_input)
            assert len(hm.get_hidden_states()) > 0

            hm.clear()
            assert len(hm.get_hidden_states()) == 0

    def test_hooks_removed_on_exit(self, fake_model, dummy_input):
        """Exiting the context manager removes all hooks."""
        with HookManager(fake_model) as hm:
            num_handles = len(hm._handles)
            assert num_handles == MOCK_NUM_LAYERS + 1  # embed + all layers

        # After exit, handles list is cleared
        assert len(hm._handles) == 0

    def test_multiple_forward_passes_accumulate(
        self, fake_model, dummy_input
    ):
        """Multiple forward passes without clear() accumulate tensors."""
        with HookManager(fake_model) as hm:
            fake_model(dummy_input)
            first_count = len(hm.get_hidden_states())

            fake_model(dummy_input)
            second_count = len(hm.get_hidden_states())

        assert second_count == 2 * first_count

    def test_clear_then_forward_captures_fresh(
        self, fake_model, dummy_input
    ):
        """After clear(), a new forward pass captures fresh tensors."""
        with HookManager(fake_model) as hm:
            fake_model(dummy_input)
            hm.clear()

            fake_model(dummy_input)
            states = hm.get_hidden_states()

        assert len(states) == MOCK_NUM_LAYERS + 1

    def test_invalid_layer_index_raises_error(self, fake_model):
        """Layer index out of range raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            HookManager(fake_model, layer_indices=[0, 99])

    def test_negative_layer_index_raises_error(self, fake_model):
        """Negative layer index raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            HookManager(fake_model, layer_indices=[-1])

    def test_no_gpt_neox_attribute_raises_error(self):
        """Model without gpt_neox attribute raises ValueError."""
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="gpt_neox"):
            HookManager(model)

    def test_single_layer_selection(self, fake_model, dummy_input):
        """Selecting a single layer works correctly."""
        with HookManager(fake_model, layer_indices=[3]) as hm:
            fake_model(dummy_input)
            states = hm.get_hidden_states()

        # 1 embedding + 1 layer = 2 tensors
        assert len(states) == 2

    def test_tensors_are_detached(self, fake_model, dummy_input):
        """Captured tensors should be detached from the computation graph."""
        with HookManager(fake_model) as hm:
            fake_model(dummy_input)
            states = hm.get_hidden_states()

        for tensor in states:
            assert not tensor.requires_grad
