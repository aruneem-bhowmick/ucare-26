"""
Tests for the model registry and specification module.

Validates YAML config loading, ModelSpec field completeness,
programmatic lookup by key, error handling for unknown models,
and the model loading interface.
"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models import (
    ModelSpec,
    get_model_spec,
    load_model,
    load_model_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_YAML = textwrap.dedent("""\
    common:
      architecture: "GPTNeoXForCausalLM"
      positional_embedding: "rotary"
      parallel_attn: true
      vocab_size: 50304
      max_positions: 2048
      revision: "main"

    models:
      pythia-70m:
        hf_id: "EleutherAI/pythia-70m-deduped"
        num_layers: 6
        hidden_size: 512
        num_heads: 8
        intermediate_size: 2048
        role: "debug"

      pythia-160m:
        hf_id: "EleutherAI/pythia-160m-deduped"
        num_layers: 12
        hidden_size: 768
        num_heads: 12
        intermediate_size: 3072
        role: "primary"
""")


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal model registry YAML to a temp file.

    Returns:
        Path to the temporary YAML file.
    """
    config_file = tmp_path / "models.yaml"
    config_file.write_text(SAMPLE_YAML)
    return config_file


# ---------------------------------------------------------------------------
# TestLoadModelRegistry
# ---------------------------------------------------------------------------


class TestLoadModelRegistry:
    """Tests for load_model_registry()."""

    def test_loads_all_models(self, sample_config: Path) -> None:
        """Registry should contain all models defined in the YAML."""
        registry = load_model_registry(sample_config)
        assert set(registry.keys()) == {"pythia-70m", "pythia-160m"}

    def test_returns_model_spec_instances(self, sample_config: Path) -> None:
        """Every value in the registry should be a ModelSpec."""
        registry = load_model_registry(sample_config)
        for spec in registry.values():
            assert isinstance(spec, ModelSpec)

    def test_common_fields_merged(self, sample_config: Path) -> None:
        """Common fields from the YAML should appear in each ModelSpec."""
        registry = load_model_registry(sample_config)
        for spec in registry.values():
            assert spec.architecture == "GPTNeoXForCausalLM"
            assert spec.positional_embedding == "rotary"
            assert spec.parallel_attn is True
            assert spec.vocab_size == 50304
            assert spec.max_positions == 2048
            assert spec.revision == "main"

    def test_per_model_fields(self, sample_config: Path) -> None:
        """Per-model fields should match the YAML entries."""
        registry = load_model_registry(sample_config)
        spec_70m = registry["pythia-70m"]
        assert spec_70m.hf_id == "EleutherAI/pythia-70m-deduped"
        assert spec_70m.num_layers == 6
        assert spec_70m.hidden_size == 512
        assert spec_70m.num_heads == 8
        assert spec_70m.intermediate_size == 2048
        assert spec_70m.role == "debug"

    def test_model_spec_is_frozen(self, sample_config: Path) -> None:
        """ModelSpec instances should be immutable."""
        registry = load_model_registry(sample_config)
        spec = registry["pythia-70m"]
        with pytest.raises(AttributeError):
            spec.hidden_size = 1024  # type: ignore[misc]

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        """Loading from a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model_registry(tmp_path / "nonexistent.yaml")

    def test_key_field_matches_dict_key(self, sample_config: Path) -> None:
        """The ModelSpec.key attribute should match its registry dict key."""
        registry = load_model_registry(sample_config)
        for key, spec in registry.items():
            assert spec.key == key


# ---------------------------------------------------------------------------
# TestGetModelSpec
# ---------------------------------------------------------------------------


class TestGetModelSpec:
    """Tests for get_model_spec()."""

    def test_valid_key(self, sample_config: Path) -> None:
        """Looking up a valid key should return the correct ModelSpec."""
        spec = get_model_spec("pythia-70m", sample_config)
        assert spec.key == "pythia-70m"
        assert spec.num_layers == 6

    def test_unknown_key_raises(self, sample_config: Path) -> None:
        """Looking up a nonexistent key should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model key"):
            get_model_spec("pythia-999b", sample_config)

    def test_error_message_lists_available(self, sample_config: Path) -> None:
        """The KeyError message should list available model keys."""
        with pytest.raises(KeyError, match="pythia-70m"):
            get_model_spec("bad-key", sample_config)


# ---------------------------------------------------------------------------
# TestModelSpecFields
# ---------------------------------------------------------------------------


class TestModelSpecFields:
    """Tests for ModelSpec field completeness and types."""

    def test_all_fields_present(self, sample_config: Path) -> None:
        """Every field in the dataclass should be populated (not None)."""
        spec = get_model_spec("pythia-70m", sample_config)
        for field_name in ModelSpec.__dataclass_fields__:
            assert getattr(spec, field_name) is not None, (
                f"Field {field_name!r} is None"
            )

    def test_numeric_fields_positive(self, sample_config: Path) -> None:
        """Numeric architecture fields should be positive integers."""
        spec = get_model_spec("pythia-70m", sample_config)
        for field_name in (
            "num_layers",
            "hidden_size",
            "num_heads",
            "intermediate_size",
            "vocab_size",
            "max_positions",
        ):
            value = getattr(spec, field_name)
            assert isinstance(value, int), f"{field_name} should be int"
            assert value > 0, f"{field_name} should be positive"

    def test_head_divides_hidden(self, sample_config: Path) -> None:
        """hidden_size should be divisible by num_heads."""
        registry = load_model_registry(sample_config)
        for spec in registry.values():
            assert spec.hidden_size % spec.num_heads == 0, (
                f"{spec.key}: hidden_size={spec.hidden_size} not divisible "
                f"by num_heads={spec.num_heads}"
            )


# ---------------------------------------------------------------------------
# TestLoadModel
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Tests for load_model()."""

    @patch("src.models.AutoTokenizer")
    @patch("src.models.AutoModelForCausalLM")
    def test_model_set_to_eval(
        self,
        mock_auto_model: MagicMock,
        mock_auto_tokenizer: MagicMock,
        sample_config: Path,
    ) -> None:
        """The loaded model should be placed in eval mode."""
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        spec = get_model_spec("pythia-70m", sample_config)
        load_model(spec)

        mock_model.eval.assert_called_once()

    @patch("src.models.AutoTokenizer")
    @patch("src.models.AutoModelForCausalLM")
    def test_tokenizer_pad_token_set(
        self,
        mock_auto_model: MagicMock,
        mock_auto_tokenizer: MagicMock,
        sample_config: Path,
    ) -> None:
        """The tokenizer pad_token should be set to the EOS token."""
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        spec = get_model_spec("pythia-70m", sample_config)
        _, tokenizer = load_model(spec)

        assert tokenizer.pad_token == "<|endoftext|>"

    @patch("src.models.AutoTokenizer")
    @patch("src.models.AutoModelForCausalLM")
    def test_tokenizer_padding_side_left(
        self,
        mock_auto_model: MagicMock,
        mock_auto_tokenizer: MagicMock,
        sample_config: Path,
    ) -> None:
        """The tokenizer should be configured for left-padding."""
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        spec = get_model_spec("pythia-70m", sample_config)
        _, tokenizer = load_model(spec)

        assert tokenizer.padding_side == "left"

    @patch("src.models.AutoTokenizer")
    @patch("src.models.AutoModelForCausalLM")
    def test_precision_passed_to_model(
        self,
        mock_auto_model: MagicMock,
        mock_auto_tokenizer: MagicMock,
        sample_config: Path,
    ) -> None:
        """The torch_dtype parameter should be forwarded to from_pretrained."""
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        spec = get_model_spec("pythia-70m", sample_config)
        load_model(spec, precision=torch.float32)

        call_kwargs = mock_auto_model.from_pretrained.call_args
        assert call_kwargs[1]["torch_dtype"] == torch.float32

    @patch("src.models.AutoTokenizer")
    @patch("src.models.AutoModelForCausalLM")
    def test_returns_model_and_tokenizer(
        self,
        mock_auto_model: MagicMock,
        mock_auto_tokenizer: MagicMock,
        sample_config: Path,
    ) -> None:
        """load_model should return a (model, tokenizer) tuple."""
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        spec = get_model_spec("pythia-70m", sample_config)
        result = load_model(spec)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("src.models.AutoTokenizer")
    @patch("src.models.AutoModelForCausalLM")
    def test_uses_correct_hf_id_and_revision(
        self,
        mock_auto_model: MagicMock,
        mock_auto_tokenizer: MagicMock,
        sample_config: Path,
    ) -> None:
        """from_pretrained should receive the hf_id and revision from the spec."""
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        spec = get_model_spec("pythia-160m", sample_config)
        load_model(spec)

        model_call = mock_auto_model.from_pretrained.call_args
        assert model_call[0][0] == "EleutherAI/pythia-160m-deduped"
        assert model_call[1]["revision"] == "main"


# ---------------------------------------------------------------------------
# TestRealConfig
# ---------------------------------------------------------------------------


class TestRealConfig:
    """Tests against the actual configs/models.yaml shipped in the repo."""

    def test_real_config_loads(self) -> None:
        """The shipped models.yaml should load without errors."""
        registry = load_model_registry()
        assert len(registry) == 5

    def test_real_config_expected_keys(self) -> None:
        """The shipped registry should contain all five Pythia checkpoints."""
        registry = load_model_registry()
        expected = {
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-2.8b",
        }
        assert set(registry.keys()) == expected

    def test_real_config_pythia_70m_spec(self) -> None:
        """The 70M spec should match the known Pythia-70M architecture."""
        spec = get_model_spec("pythia-70m")
        assert spec.hf_id == "EleutherAI/pythia-70m-deduped"
        assert spec.num_layers == 6
        assert spec.hidden_size == 512
        assert spec.num_heads == 8
        assert spec.intermediate_size == 2048

    def test_real_config_pythia_2_8b_spec(self) -> None:
        """The 2.8B spec should match the known Pythia-2.8B architecture."""
        spec = get_model_spec("pythia-2.8b")
        assert spec.hf_id == "EleutherAI/pythia-2.8b-deduped"
        assert spec.num_layers == 32
        assert spec.hidden_size == 2560
        assert spec.num_heads == 32
        assert spec.intermediate_size == 10240

    def test_all_models_have_complete_fields(self) -> None:
        """Every model in the real config should have all fields populated."""
        registry = load_model_registry()
        for key, spec in registry.items():
            for field_name in ModelSpec.__dataclass_fields__:
                assert getattr(spec, field_name) is not None, (
                    f"Model {key!r}: field {field_name!r} is None"
                )
