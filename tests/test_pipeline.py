"""
Tests for the end-to-end extraction pipeline orchestrator.

All tests are fully mocked — no network access, model downloads,
or GPU required. Validates that ``ExtractionPipeline`` correctly
wires together model loading, data loading, hook-based extraction,
pooling, and caching.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.extraction.pipeline import ExtractionPipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOCK_NUM_LAYERS = 6
MOCK_HIDDEN_SIZE = 32
MOCK_SEQ_LEN = 8
MOCK_BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self._hidden_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **kwargs):
        batch, seq = input_ids.shape
        return torch.randn(batch, seq, self._hidden_size)


class FakeGPTNeoXLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self._hidden_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, **kwargs):
        return (torch.randn_like(hidden_states),)


class FakeGPTNeoXModel(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.embed_in = FakeEmbedding(hidden_size)
        self.layers = nn.ModuleList(
            [FakeGPTNeoXLayer(hidden_size) for _ in range(num_layers)]
        )


class FakeModel(nn.Module):
    """Minimal GPTNeoX model for pipeline testing."""

    def __init__(
        self,
        num_layers=MOCK_NUM_LAYERS,
        hidden_size=MOCK_HIDDEN_SIZE,
    ):
        super().__init__()
        self.gpt_neox = FakeGPTNeoXModel(num_layers, hidden_size)
        self.config = MagicMock()
        self.config.num_hidden_layers = num_layers
        self.config.hidden_size = hidden_size

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.gpt_neox.embed_in(input_ids)
        for layer in self.gpt_neox.layers:
            hidden = layer(hidden)[0]
        return hidden


def _make_fake_dataset(num_examples=4):
    """Create a list-of-dicts mock dataset."""
    data = []
    for i in range(num_examples):
        data.append({
            "text": f"This is example number {i}.",
            "label": i % 2,
            "example_id": i,
        })

    # Make it behave like a HuggingFace Dataset
    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(data)
    mock_ds.__getitem__ = lambda self, idx: data[idx]
    return mock_ds


def _make_fake_tokenizer():
    """Create a tokenizer mock that returns proper tensors."""
    tokenizer = MagicMock()
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = "left"

    def tokenize_fn(texts, **kwargs):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.randint(0, 100, (batch_size, MOCK_SEQ_LEN)),
            "attention_mask": torch.ones(batch_size, MOCK_SEQ_LEN, dtype=torch.long),
        }

    tokenizer.side_effect = tokenize_fn
    return tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extraction_config(tmp_path):
    """Write a minimal extraction.yaml and return its path."""
    config_path = tmp_path / "extraction.yaml"
    config_path.write_text(
        "extraction:\n"
        "  pooling: last_token\n"
        "  layers: -1\n"
        "  batch_size: 2\n"
        "  max_seq_length: 64\n"
        f"  output_dir: '{tmp_path.as_posix()}/outputs'\n"
        "  seed: 42\n"
        "  precision: float16\n"
        "  cache_format: safetensors\n"
    )
    return config_path


@pytest.fixture
def mock_model_spec():
    """Create a ModelSpec-like object."""
    spec = MagicMock()
    spec.key = "pythia-70m"
    spec.hf_id = "EleutherAI/pythia-70m-deduped"
    spec.num_layers = MOCK_NUM_LAYERS
    spec.hidden_size = MOCK_HIDDEN_SIZE
    spec.revision = "main"
    return spec


@pytest.fixture
def mock_task_spec():
    """Create a TaskSpec-like object."""
    spec = MagicMock()
    spec.name = "sst2"
    spec.family = "shallow_semantic"
    spec.label_type = "classification"
    spec.extraction_position = "last_token"
    return spec


# ---------------------------------------------------------------------------
# TestExtractionPipeline
# ---------------------------------------------------------------------------


class TestExtractionPipeline:
    """Tests for ExtractionPipeline (all mocked)."""

    @patch("src.extraction.pipeline.get_task_spec")
    @patch("src.extraction.pipeline.get_model_spec")
    def test_loads_correct_specs(
        self,
        mock_get_model,
        mock_get_task,
        mock_model_spec,
        mock_task_spec,
        extraction_config,
    ):
        """Pipeline loads the correct ModelSpec and TaskSpec."""
        mock_get_model.return_value = mock_model_spec
        mock_get_task.return_value = mock_task_spec

        pipeline = ExtractionPipeline(
            "pythia-70m", "sst2", config_path=extraction_config
        )

        mock_get_model.assert_called_once_with("pythia-70m")
        mock_get_task.assert_called_once_with("sst2")
        assert pipeline.model_spec is mock_model_spec
        assert pipeline.task_spec is mock_task_spec

    @patch("src.extraction.pipeline.get_task_spec")
    @patch("src.extraction.pipeline.get_model_spec")
    @patch("src.extraction.pipeline.load_model")
    def test_run_creates_output_directory(
        self,
        mock_load_model,
        mock_get_model,
        mock_get_task,
        mock_model_spec,
        mock_task_spec,
        extraction_config,
        tmp_path,
    ):
        """run() creates the output directory and returns a cache path."""
        mock_get_model.return_value = mock_model_spec
        mock_get_task.return_value = mock_task_spec

        fake_model = FakeModel()
        fake_model.eval()
        fake_tokenizer = _make_fake_tokenizer()
        mock_load_model.return_value = (fake_model, fake_tokenizer)

        pipeline = ExtractionPipeline(
            "pythia-70m", "sst2", config_path=extraction_config
        )

        with patch.object(pipeline, "_load_dataset", return_value=_make_fake_dataset(4)):
            cache_dir = pipeline.run(split="validation", max_examples=4)

        assert cache_dir.exists()
        assert isinstance(cache_dir, Path)

    @patch("src.extraction.pipeline.get_task_spec")
    @patch("src.extraction.pipeline.get_model_spec")
    @patch("src.extraction.pipeline.load_model")
    def test_run_produces_manifest(
        self,
        mock_load_model,
        mock_get_model,
        mock_get_task,
        mock_model_spec,
        mock_task_spec,
        extraction_config,
        tmp_path,
    ):
        """run() produces a manifest.jsonl with correct metadata."""
        mock_get_model.return_value = mock_model_spec
        mock_get_task.return_value = mock_task_spec

        fake_model = FakeModel()
        fake_model.eval()
        fake_tokenizer = _make_fake_tokenizer()
        mock_load_model.return_value = (fake_model, fake_tokenizer)

        pipeline = ExtractionPipeline(
            "pythia-70m", "sst2", config_path=extraction_config
        )

        num_examples = 4
        with patch.object(
            pipeline, "_load_dataset", return_value=_make_fake_dataset(num_examples)
        ):
            cache_dir = pipeline.run(split="validation", max_examples=num_examples)

        manifest_path = cache_dir / "manifest.jsonl"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) == num_examples
        assert entries[0]["model_key"] == "pythia-70m"
        assert entries[0]["dataset_name"] == "sst2"
        assert entries[0]["pool_strategy"] == "last_token"
        assert entries[0]["split"] == "validation"

    @patch("src.extraction.pipeline.get_task_spec")
    @patch("src.extraction.pipeline.get_model_spec")
    @patch("src.extraction.pipeline.load_model")
    def test_run_produces_safetensors_files(
        self,
        mock_load_model,
        mock_get_model,
        mock_get_task,
        mock_model_spec,
        mock_task_spec,
        extraction_config,
        tmp_path,
    ):
        """run() produces safetensors files for each captured layer."""
        mock_get_model.return_value = mock_model_spec
        mock_get_task.return_value = mock_task_spec

        fake_model = FakeModel()
        fake_model.eval()
        fake_tokenizer = _make_fake_tokenizer()
        mock_load_model.return_value = (fake_model, fake_tokenizer)

        pipeline = ExtractionPipeline(
            "pythia-70m", "sst2", config_path=extraction_config
        )

        with patch.object(pipeline, "_load_dataset", return_value=_make_fake_dataset(4)):
            cache_dir = pipeline.run(split="validation", max_examples=4)

        # Should have safetensors files (embedding + all layers)
        safetensor_files = list(cache_dir.glob("layer_*.safetensors"))
        assert len(safetensor_files) > 0

    @patch("src.extraction.pipeline.get_task_spec")
    @patch("src.extraction.pipeline.get_model_spec")
    @patch("src.extraction.pipeline.load_model")
    def test_run_returns_cache_directory_path(
        self,
        mock_load_model,
        mock_get_model,
        mock_get_task,
        mock_model_spec,
        mock_task_spec,
        extraction_config,
        tmp_path,
    ):
        """run() returns the cache directory as a Path object."""
        mock_get_model.return_value = mock_model_spec
        mock_get_task.return_value = mock_task_spec

        fake_model = FakeModel()
        fake_model.eval()
        fake_tokenizer = _make_fake_tokenizer()
        mock_load_model.return_value = (fake_model, fake_tokenizer)

        pipeline = ExtractionPipeline(
            "pythia-70m", "sst2", config_path=extraction_config
        )

        with patch.object(pipeline, "_load_dataset", return_value=_make_fake_dataset(4)):
            result = pipeline.run(split="validation", max_examples=4)

        assert isinstance(result, Path)
        # Should be {output_dir}/pythia-70m/sst2
        assert result.name == "sst2"
        assert result.parent.name == "pythia-70m"

    @patch("src.extraction.pipeline.get_task_spec")
    @patch("src.extraction.pipeline.get_model_spec")
    def test_config_parsing(
        self,
        mock_get_model,
        mock_get_task,
        mock_model_spec,
        mock_task_spec,
        extraction_config,
    ):
        """Pipeline correctly parses extraction config values."""
        mock_get_model.return_value = mock_model_spec
        mock_get_task.return_value = mock_task_spec

        pipeline = ExtractionPipeline(
            "pythia-70m", "sst2", config_path=extraction_config
        )

        assert pipeline.batch_size == 2
        assert pipeline.max_seq_length == 64
        assert pipeline.pool_strategy == "last_token"
        assert pipeline.seed == 42
        assert pipeline.layer_indices is None  # -1 means all layers
