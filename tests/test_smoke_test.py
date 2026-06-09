"""
Tests for src.extraction.smoke_test module.

Tests are organized into six groups covering all public functions:

1. ``TestSetSeed`` -- Reproducibility and seed isolation
2. ``TestValidateHiddenStates`` -- Shape and structure assertions
3. ``TestGetLastNonPaddingRepresentation`` -- Last-token pooling logic
4. ``TestGetMeanPooledRepresentation`` -- Mean pooling logic
5. ``TestLoadModelAndTokenizer`` -- Model/tokenizer loading (mocked)
6. ``TestRunSmokeTest`` -- Integration/orchestration tests (mocked)

All tests use mock objects or synthetic tensors so that no model
downloads, GPU access, or network connectivity is required.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, call

from src.extraction.smoke_test import (
    set_seed,
    load_model_and_tokenizer,
    extract_hidden_states,
    validate_hidden_states,
    get_last_non_padding_representation,
    get_mean_pooled_representation,
    run_smoke_test,
    DEFAULT_MODEL_NAME,
    DEFAULT_REVISION,
    DEFAULT_SEED,
)
from tests.conftest import (
    MOCK_NUM_LAYERS,
    MOCK_HIDDEN_SIZE,
    MOCK_SEQ_LEN,
    MOCK_BATCH_SIZE,
)


# ===================================================================
# TestSetSeed
# ===================================================================


class TestSetSeed:
    """Tests for the set_seed function."""

    def test_deterministic_output(self):
        """Setting the same seed twice produces identical random tensors."""
        set_seed(99)
        a = torch.randn(5)
        set_seed(99)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        """Setting different seeds produces different random tensors."""
        set_seed(1)
        a = torch.randn(5)
        set_seed(2)
        b = torch.randn(5)
        assert not torch.equal(a, b)

    def test_default_seed_value(self):
        """The default seed parameter matches the module constant."""
        # Verify that calling set_seed() without arguments uses DEFAULT_SEED
        set_seed()
        a = torch.randn(3)
        set_seed(DEFAULT_SEED)
        b = torch.randn(3)
        assert torch.equal(a, b)


# ===================================================================
# TestValidateHiddenStates
# ===================================================================


class TestValidateHiddenStates:
    """Tests for the validate_hidden_states function."""

    def test_valid_hidden_states(self, mock_hidden_states):
        """Validation passes with correct structure (7 tensors, hidden_size=512)."""
        # Should not raise
        validate_hidden_states(
            mock_hidden_states, MOCK_NUM_LAYERS, MOCK_HIDDEN_SIZE
        )

    def test_wrong_num_layers_raises(self):
        """Validation raises AssertionError when layer count is wrong."""
        # Create 5 tensors but claim there should be 6 layers (= 7 tensors)
        bad_states = tuple(
            torch.randn(1, MOCK_SEQ_LEN, MOCK_HIDDEN_SIZE) for _ in range(5)
        )
        with pytest.raises(AssertionError, match="Expected 7"):
            validate_hidden_states(bad_states, MOCK_NUM_LAYERS, MOCK_HIDDEN_SIZE)

    def test_wrong_hidden_size_raises(self, mock_hidden_states):
        """Validation raises AssertionError when hidden_size mismatches."""
        with pytest.raises(AssertionError, match="dimension"):
            validate_hidden_states(
                mock_hidden_states, MOCK_NUM_LAYERS, 256  # wrong size
            )

    def test_empty_tuple_raises(self):
        """Validation raises AssertionError for empty hidden states."""
        with pytest.raises(AssertionError):
            validate_hidden_states((), MOCK_NUM_LAYERS, MOCK_HIDDEN_SIZE)

    def test_single_layer_model(self):
        """Validation passes for a model with a single transformer block."""
        states = tuple(
            torch.randn(1, MOCK_SEQ_LEN, 64) for _ in range(2)  # embed + 1 block
        )
        validate_hidden_states(states, expected_num_layers=1, expected_hidden_size=64)


# ===================================================================
# TestGetLastNonPaddingRepresentation
# ===================================================================


class TestGetLastNonPaddingRepresentation:
    """Tests for the get_last_non_padding_representation function."""

    def test_output_shape(self, mock_hidden_states, mock_attention_mask):
        """Output shape should be (batch_size, hidden_size)."""
        result = get_last_non_padding_representation(
            mock_hidden_states, mock_attention_mask
        )
        assert result.shape == (MOCK_BATCH_SIZE, MOCK_HIDDEN_SIZE)

    def test_correct_token_selected_with_padding(self):
        """With left padding, the last non-pad token is the final position."""
        # Create a simple hidden state where each position has a unique value
        hidden = torch.arange(4 * 3, dtype=torch.float).reshape(1, 4, 3)
        # Wrap in a tuple (simulating single-layer output)
        hidden_states = (hidden,)
        # Left padding: first 2 tokens are padding
        mask = torch.tensor([[0, 0, 1, 1]])
        result = get_last_non_padding_representation(hidden_states, mask, layer_idx=0)
        # Last non-padding token is at position 3
        expected = hidden[0, 3, :]
        assert torch.equal(result.squeeze(), expected)

    def test_no_padding(self, mock_hidden_states, mock_attention_mask_no_padding):
        """With no padding, the last token is the final position."""
        result = get_last_non_padding_representation(
            mock_hidden_states, mock_attention_mask_no_padding
        )
        # The last token should be at position seq_len - 1
        expected = mock_hidden_states[-1][0, MOCK_SEQ_LEN - 1, :]
        assert torch.equal(result.squeeze(), expected)

    def test_layer_idx_selects_correct_layer(
        self, mock_hidden_states, mock_attention_mask
    ):
        """Specifying layer_idx=0 should use the embedding layer output."""
        result_embed = get_last_non_padding_representation(
            mock_hidden_states, mock_attention_mask, layer_idx=0
        )
        result_final = get_last_non_padding_representation(
            mock_hidden_states, mock_attention_mask, layer_idx=-1
        )
        # Embedding layer and final layer should differ (random tensors)
        assert not torch.equal(result_embed, result_final)

    def test_batch_of_two(self):
        """Test with a batch of two sequences with different padding amounts."""
        batch_size = 2
        seq_len = 6
        hidden_size = 4

        hidden = torch.arange(
            batch_size * seq_len * hidden_size, dtype=torch.float
        ).reshape(batch_size, seq_len, hidden_size)
        hidden_states = (hidden,)

        # Sequence 0: 1 padding token (left), last real at position 5
        # Sequence 1: 3 padding tokens (left), last real at position 5
        mask = torch.tensor([
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ])

        result = get_last_non_padding_representation(
            hidden_states, mask, layer_idx=0
        )
        assert result.shape == (batch_size, hidden_size)
        # Both should select position 5 (last position)
        assert torch.equal(result[0], hidden[0, 5, :])
        assert torch.equal(result[1], hidden[1, 5, :])


# ===================================================================
# TestGetMeanPooledRepresentation
# ===================================================================


class TestGetMeanPooledRepresentation:
    """Tests for the get_mean_pooled_representation function."""

    def test_output_shape(self, mock_hidden_states, mock_attention_mask):
        """Output shape should be (batch_size, hidden_size)."""
        result = get_mean_pooled_representation(
            mock_hidden_states, mock_attention_mask
        )
        assert result.shape == (MOCK_BATCH_SIZE, MOCK_HIDDEN_SIZE)

    def test_mean_excludes_padding(self):
        """Mean pooling should only average over non-padding positions."""
        # 1 batch, 4 positions, 2 features
        hidden = torch.tensor([[[1.0, 2.0],
                                [3.0, 4.0],
                                [5.0, 6.0],
                                [7.0, 8.0]]])
        hidden_states = (hidden,)
        # First 2 positions are padding
        mask = torch.tensor([[0, 0, 1, 1]])

        result = get_mean_pooled_representation(hidden_states, mask, layer_idx=0)
        # Mean of positions 2 and 3: [(5+7)/2, (6+8)/2] = [6.0, 7.0]
        expected = torch.tensor([[6.0, 7.0]])
        assert torch.allclose(result, expected)

    def test_no_padding_equals_full_mean(
        self, mock_hidden_states, mock_attention_mask_no_padding
    ):
        """With no padding, mean pooling equals a simple mean over seq_len."""
        result = get_mean_pooled_representation(
            mock_hidden_states, mock_attention_mask_no_padding
        )
        # Compute expected mean directly
        expected = mock_hidden_states[-1].float().mean(dim=1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_single_non_padding_token(self):
        """With only one real token, mean pooling equals that token's repr."""
        hidden = torch.tensor([[[10.0, 20.0],
                                [30.0, 40.0],
                                [50.0, 60.0]]])
        hidden_states = (hidden,)
        # Only the last position is real
        mask = torch.tensor([[0, 0, 1]])

        result = get_mean_pooled_representation(hidden_states, mask, layer_idx=0)
        expected = torch.tensor([[50.0, 60.0]])
        assert torch.allclose(result, expected)

    def test_layer_idx_selects_correct_layer(
        self, mock_hidden_states, mock_attention_mask
    ):
        """Specifying layer_idx=0 should use the embedding layer output."""
        result_embed = get_mean_pooled_representation(
            mock_hidden_states, mock_attention_mask, layer_idx=0
        )
        result_final = get_mean_pooled_representation(
            mock_hidden_states, mock_attention_mask, layer_idx=-1
        )
        assert not torch.equal(result_embed, result_final)

    def test_batch_of_two_different_padding(self):
        """Mean pooling with a batch of two, each with different padding."""
        hidden = torch.tensor([
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
        ])
        hidden_states = (hidden,)
        # Seq 0: 1 pad token, real tokens at positions 1-3
        # Seq 1: 2 pad tokens, real tokens at positions 2-3
        mask = torch.tensor([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ])

        result = get_mean_pooled_representation(hidden_states, mask, layer_idx=0)
        # Seq 0: mean of [2,2], [3,3], [4,4] = [3, 3]
        # Seq 1: mean of [7,7], [8,8] = [7.5, 7.5]
        expected = torch.tensor([[3.0, 3.0], [7.5, 7.5]])
        assert torch.allclose(result, expected)


# ===================================================================
# TestLoadModelAndTokenizer
# ===================================================================


class TestLoadModelAndTokenizer:
    """Tests for model/tokenizer loading (fully mocked)."""

    @patch("src.extraction.smoke_test.AutoTokenizer")
    @patch("src.extraction.smoke_test.AutoModelForCausalLM")
    def test_model_loaded_in_eval_mode(self, mock_auto_model, mock_auto_tok):
        """Model should be set to eval mode after loading."""
        model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = model_instance

        tok_instance = MagicMock()
        tok_instance.eos_token = "<|endoftext|>"
        mock_auto_tok.from_pretrained.return_value = tok_instance

        model, _ = load_model_and_tokenizer()
        model.eval.assert_called_once()

    @patch("src.extraction.smoke_test.AutoTokenizer")
    @patch("src.extraction.smoke_test.AutoModelForCausalLM")
    def test_tokenizer_pad_token_set(self, mock_auto_model, mock_auto_tok):
        """Tokenizer pad_token should be set to eos_token."""
        model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = model_instance

        tok_instance = MagicMock()
        tok_instance.eos_token = "<|endoftext|>"
        mock_auto_tok.from_pretrained.return_value = tok_instance

        _, tokenizer = load_model_and_tokenizer()
        assert tokenizer.pad_token == "<|endoftext|>"

    @patch("src.extraction.smoke_test.AutoTokenizer")
    @patch("src.extraction.smoke_test.AutoModelForCausalLM")
    def test_tokenizer_padding_side_left(self, mock_auto_model, mock_auto_tok):
        """Tokenizer padding_side should be 'left'."""
        model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = model_instance

        tok_instance = MagicMock()
        tok_instance.eos_token = "<|endoftext|>"
        mock_auto_tok.from_pretrained.return_value = tok_instance

        _, tokenizer = load_model_and_tokenizer()
        assert tokenizer.padding_side == "left"

    @patch("src.extraction.smoke_test.AutoTokenizer")
    @patch("src.extraction.smoke_test.AutoModelForCausalLM")
    def test_model_precision_passed(self, mock_auto_model, mock_auto_tok):
        """Model should be loaded with the specified torch_dtype."""
        model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = model_instance

        tok_instance = MagicMock()
        tok_instance.eos_token = "<|endoftext|>"
        mock_auto_tok.from_pretrained.return_value = tok_instance

        load_model_and_tokenizer(precision=torch.float32)
        mock_auto_model.from_pretrained.assert_called_once_with(
            DEFAULT_MODEL_NAME,
            revision=DEFAULT_REVISION,
            torch_dtype=torch.float32,
        )


# ===================================================================
# TestExtractHiddenStates
# ===================================================================


class TestExtractHiddenStates:
    """Tests for the extract_hidden_states function."""

    def test_returns_hidden_states_and_mask(self, mock_model, mock_tokenizer):
        """Function should return a (hidden_states, attention_mask) tuple."""
        hidden_states, attention_mask = extract_hidden_states(
            mock_model, mock_tokenizer, "test"
        )
        assert isinstance(hidden_states, tuple)
        assert isinstance(attention_mask, torch.Tensor)

    def test_calls_model_with_output_hidden_states(self, mock_model, mock_tokenizer):
        """Model should be called with output_hidden_states=True."""
        extract_hidden_states(mock_model, mock_tokenizer, "test")
        # The model was called; check that output_hidden_states was passed
        _, kwargs = mock_model.call_args
        assert kwargs.get("output_hidden_states") is True


# ===================================================================
# TestRunSmokeTest
# ===================================================================


class TestRunSmokeTest:
    """Integration tests for the full smoke test pipeline (mocked)."""

    @patch("src.extraction.smoke_test.load_model_and_tokenizer")
    @patch("src.extraction.smoke_test.extract_hidden_states")
    def test_returns_expected_keys(self, mock_extract, mock_load):
        """run_smoke_test should return a dict with the expected keys."""
        # Set up mock model with config
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = MOCK_NUM_LAYERS
        mock_model.config.hidden_size = MOCK_HIDDEN_SIZE
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Set up mock hidden states
        gen = torch.Generator().manual_seed(0)
        hs = tuple(
            torch.randn(1, MOCK_SEQ_LEN, MOCK_HIDDEN_SIZE, generator=gen)
            for _ in range(MOCK_NUM_LAYERS + 1)
        )
        mask = torch.ones(1, MOCK_SEQ_LEN, dtype=torch.long)
        mock_extract.return_value = (hs, mask)

        results = run_smoke_test()
        expected_keys = {
            "hidden_states",
            "attention_mask",
            "last_token_repr",
            "mean_pooled_repr",
            "num_layers",
            "hidden_size",
        }
        assert set(results.keys()) == expected_keys

    @patch("src.extraction.smoke_test.set_seed")
    @patch("src.extraction.smoke_test.load_model_and_tokenizer")
    @patch("src.extraction.smoke_test.extract_hidden_states")
    def test_calls_set_seed(self, mock_extract, mock_load, mock_set_seed):
        """run_smoke_test should call set_seed for reproducibility."""
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = MOCK_NUM_LAYERS
        mock_model.config.hidden_size = MOCK_HIDDEN_SIZE
        mock_load.return_value = (mock_model, MagicMock())

        gen = torch.Generator().manual_seed(0)
        hs = tuple(
            torch.randn(1, MOCK_SEQ_LEN, MOCK_HIDDEN_SIZE, generator=gen)
            for _ in range(MOCK_NUM_LAYERS + 1)
        )
        mask = torch.ones(1, MOCK_SEQ_LEN, dtype=torch.long)
        mock_extract.return_value = (hs, mask)

        run_smoke_test(seed=123)
        mock_set_seed.assert_called_once_with(123)

    @patch("src.extraction.smoke_test.load_model_and_tokenizer")
    @patch("src.extraction.smoke_test.extract_hidden_states")
    def test_num_layers_matches_hidden_states(self, mock_extract, mock_load):
        """The returned num_layers should match the hidden states tuple length."""
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = MOCK_NUM_LAYERS
        mock_model.config.hidden_size = MOCK_HIDDEN_SIZE
        mock_load.return_value = (mock_model, MagicMock())

        gen = torch.Generator().manual_seed(0)
        hs = tuple(
            torch.randn(1, MOCK_SEQ_LEN, MOCK_HIDDEN_SIZE, generator=gen)
            for _ in range(MOCK_NUM_LAYERS + 1)
        )
        mask = torch.ones(1, MOCK_SEQ_LEN, dtype=torch.long)
        mock_extract.return_value = (hs, mask)

        results = run_smoke_test()
        assert results["num_layers"] == MOCK_NUM_LAYERS + 1

    @patch("src.extraction.smoke_test.load_model_and_tokenizer")
    @patch("src.extraction.smoke_test.extract_hidden_states")
    def test_hidden_size_matches_config(self, mock_extract, mock_load):
        """The returned hidden_size should match the model config."""
        mock_model = MagicMock()
        mock_model.config.num_hidden_layers = MOCK_NUM_LAYERS
        mock_model.config.hidden_size = MOCK_HIDDEN_SIZE
        mock_load.return_value = (mock_model, MagicMock())

        gen = torch.Generator().manual_seed(0)
        hs = tuple(
            torch.randn(1, MOCK_SEQ_LEN, MOCK_HIDDEN_SIZE, generator=gen)
            for _ in range(MOCK_NUM_LAYERS + 1)
        )
        mask = torch.ones(1, MOCK_SEQ_LEN, dtype=torch.long)
        mock_extract.return_value = (hs, mask)

        results = run_smoke_test()
        assert results["hidden_size"] == MOCK_HIDDEN_SIZE
