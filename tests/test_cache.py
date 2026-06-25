"""
Tests for fp16 caching with safetensors and JSONL manifest.

Validates save/load round-trips, manifest structure, fp16 precision,
and verification of cache integrity.
"""

import json

import pytest
import torch

from src.extraction.cache import (
    load_representations,
    save_representations,
    verify_manifest,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_EXAMPLES = 5
HIDDEN_SIZE = 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_representations():
    """Create sample representations for two layers."""
    gen = torch.Generator().manual_seed(42)
    return {
        0: torch.randn(NUM_EXAMPLES, HIDDEN_SIZE, generator=gen),
        3: torch.randn(NUM_EXAMPLES, HIDDEN_SIZE, generator=gen),
    }


@pytest.fixture
def sample_metadata():
    """Create sample per-example metadata."""
    return {
        "labels": [0, 1, 0, 1, 0],
        "example_ids": [10, 20, 30, 40, 50],
        "token_counts": [5, 8, 3, 12, 7],
    }


@pytest.fixture
def saved_cache_dir(tmp_path, sample_representations, sample_metadata):
    """Save representations and return the cache directory."""
    cache_dir = save_representations(
        representations=sample_representations,
        labels=sample_metadata["labels"],
        example_ids=sample_metadata["example_ids"],
        token_counts=sample_metadata["token_counts"],
        output_dir=str(tmp_path),
        model_key="pythia-70m",
        dataset_name="sst2",
        split="validation",
        pool_strategy="last_token",
        seed=42,
        model_revision="main",
    )
    return cache_dir


# ---------------------------------------------------------------------------
# TestSaveRepresentations
# ---------------------------------------------------------------------------


class TestSaveRepresentations:
    """Tests for save_representations."""

    def test_creates_correct_directory_structure(
        self, tmp_path, sample_representations, sample_metadata
    ):
        """Creates {output_dir}/{model_key}/{dataset_name}/ directory."""
        cache_dir = save_representations(
            representations=sample_representations,
            labels=sample_metadata["labels"],
            example_ids=sample_metadata["example_ids"],
            token_counts=sample_metadata["token_counts"],
            output_dir=str(tmp_path),
            model_key="pythia-70m",
            dataset_name="sst2",
            split="validation",
            pool_strategy="last_token",
            seed=42,
        )
        assert cache_dir.exists()
        assert cache_dir == tmp_path / "pythia-70m" / "sst2"

    def test_writes_safetensors_per_layer(self, saved_cache_dir):
        """Creates one safetensors file per layer."""
        assert (saved_cache_dir / "layer_00.safetensors").exists()
        assert (saved_cache_dir / "layer_03.safetensors").exists()

    def test_writes_manifest_jsonl(self, saved_cache_dir):
        """Creates a manifest.jsonl file."""
        manifest_path = saved_cache_dir / "manifest.jsonl"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == NUM_EXAMPLES

    def test_tensors_saved_in_fp16(self, saved_cache_dir):
        """Saved tensors are in float16 precision."""
        from safetensors.torch import load_file

        data = load_file(str(saved_cache_dir / "layer_00.safetensors"))
        assert data["representations"].dtype == torch.float16

    def test_manifest_contains_expected_fields(self, saved_cache_dir):
        """Each manifest entry has the required metadata fields."""
        manifest_path = saved_cache_dir / "manifest.jsonl"
        with open(manifest_path) as f:
            entry = json.loads(f.readline())

        expected_fields = {
            "example_id",
            "label",
            "token_count",
            "layer_indices",
            "pool_strategy",
            "model_key",
            "model_revision",
            "dataset_name",
            "split",
            "seed",
            "torch_version",
        }
        assert expected_fields.issubset(set(entry.keys()))

    def test_manifest_metadata_values(self, saved_cache_dir):
        """Manifest entries contain correct metadata values."""
        manifest_path = saved_cache_dir / "manifest.jsonl"
        with open(manifest_path) as f:
            entry = json.loads(f.readline())

        assert entry["model_key"] == "pythia-70m"
        assert entry["dataset_name"] == "sst2"
        assert entry["split"] == "validation"
        assert entry["pool_strategy"] == "last_token"
        assert entry["seed"] == 42
        assert entry["layer_indices"] == [0, 3]


# ---------------------------------------------------------------------------
# TestLoadRepresentations
# ---------------------------------------------------------------------------


class TestLoadRepresentations:
    """Tests for load_representations."""

    def test_round_trip_tensor_values(
        self, saved_cache_dir, sample_representations
    ):
        """Save→load produces tensors that match (within fp16 tolerance)."""
        loaded_repr, _ = load_representations(saved_cache_dir)

        assert set(loaded_repr.keys()) == set(sample_representations.keys())
        for layer_idx in sample_representations:
            expected = sample_representations[layer_idx].half()
            actual = loaded_repr[layer_idx]
            assert torch.allclose(actual, expected)

    def test_round_trip_manifest_entries(
        self, saved_cache_dir, sample_metadata
    ):
        """Save→load produces correct manifest entries."""
        _, manifest = load_representations(saved_cache_dir)

        assert len(manifest) == NUM_EXAMPLES
        assert manifest[0]["example_id"] == sample_metadata["example_ids"][0]
        assert manifest[0]["label"] == sample_metadata["labels"][0]

    def test_loaded_tensor_shapes(self, saved_cache_dir):
        """Loaded tensors have shape (num_examples, hidden)."""
        loaded_repr, _ = load_representations(saved_cache_dir)
        for tensor in loaded_repr.values():
            assert tensor.shape == (NUM_EXAMPLES, HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# TestVerifyManifest
# ---------------------------------------------------------------------------


class TestVerifyManifest:
    """Tests for verify_manifest."""

    def test_valid_cache_passes(self, saved_cache_dir):
        """A correctly saved cache passes verification."""
        assert verify_manifest(saved_cache_dir) is True

    def test_missing_layer_file_raises(self, saved_cache_dir):
        """Missing a layer file raises ValueError."""
        # Delete one of the layer files
        (saved_cache_dir / "layer_00.safetensors").unlink()

        with pytest.raises(ValueError, match="Layer file missing"):
            verify_manifest(saved_cache_dir)

    def test_shape_mismatch_raises(self, saved_cache_dir):
        """Tensor with wrong number of examples raises ValueError."""
        from safetensors.torch import save_file

        # Overwrite layer_00 with a tensor that has wrong example count
        wrong_tensor = torch.randn(NUM_EXAMPLES + 5, HIDDEN_SIZE).half()
        save_file(
            {"representations": wrong_tensor},
            str(saved_cache_dir / "layer_00.safetensors"),
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            verify_manifest(saved_cache_dir)

    def test_missing_manifest_raises(self, tmp_path):
        """Missing manifest.jsonl raises ValueError."""
        with pytest.raises(ValueError, match="Manifest file not found"):
            verify_manifest(tmp_path)

    def test_empty_manifest_raises(self, saved_cache_dir):
        """Empty manifest.jsonl raises ValueError."""
        manifest_path = saved_cache_dir / "manifest.jsonl"
        manifest_path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            verify_manifest(saved_cache_dir)
