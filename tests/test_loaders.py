"""
Tests for HuggingFace dataset loaders.

All tests mock the ``datasets.load_dataset`` call to avoid network
access during testing. Each test validates the standardized output
schema (``text``, ``label``, ``example_id``).
"""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from src.data.loaders import load_lama_trex, load_mrpc, load_sst2


# ---------------------------------------------------------------------------
# SST-2 loader tests
# ---------------------------------------------------------------------------


def _make_mock_sst2(num_examples: int = 10) -> Dataset:
    """Create a mock SST-2 dataset matching the HF schema."""
    return Dataset.from_dict({
        "sentence": [f"This is sentence {i}." for i in range(num_examples)],
        "label": [i % 2 for i in range(num_examples)],
        "idx": list(range(num_examples)),
    })


class TestLoadSst2:
    """Tests for load_sst2()."""

    @patch("src.data.loaders.load_dataset")
    def test_correct_columns(self, mock_load: MagicMock) -> None:
        """Output dataset should have text, label, example_id columns."""
        mock_load.return_value = _make_mock_sst2()
        ds = load_sst2()
        assert "text" in ds.column_names
        assert "label" in ds.column_names
        assert "example_id" in ds.column_names
        assert "sentence" not in ds.column_names
        assert "idx" not in ds.column_names

    @patch("src.data.loaders.load_dataset")
    def test_label_range(self, mock_load: MagicMock) -> None:
        """Labels should be 0 or 1."""
        mock_load.return_value = _make_mock_sst2()
        ds = load_sst2()
        assert all(label in (0, 1) for label in ds["label"])

    @patch("src.data.loaders.load_dataset")
    def test_text_values(self, mock_load: MagicMock) -> None:
        """Text column should contain the original sentences."""
        mock_load.return_value = _make_mock_sst2()
        ds = load_sst2()
        assert ds["text"][0] == "This is sentence 0."

    @patch("src.data.loaders.load_dataset")
    def test_subsampling(self, mock_load: MagicMock) -> None:
        """Providing max_examples should reduce dataset size."""
        mock_load.return_value = _make_mock_sst2(20)
        ds = load_sst2(max_examples=5)
        assert len(ds) == 5

    @patch("src.data.loaders.load_dataset")
    def test_no_subsampling_when_larger(self, mock_load: MagicMock) -> None:
        """When max_examples >= len(ds), all examples should be kept."""
        mock_load.return_value = _make_mock_sst2(5)
        ds = load_sst2(max_examples=100)
        assert len(ds) == 5

    @patch("src.data.loaders.load_dataset")
    def test_split_passed_to_load_dataset(self, mock_load: MagicMock) -> None:
        """The split argument should be forwarded to load_dataset."""
        mock_load.return_value = _make_mock_sst2()
        load_sst2(split="train")
        mock_load.assert_called_once_with("glue", "sst2", split="train")


# ---------------------------------------------------------------------------
# MRPC loader tests
# ---------------------------------------------------------------------------


def _make_mock_mrpc(num_examples: int = 10) -> Dataset:
    """Create a mock MRPC dataset matching the HF schema."""
    return Dataset.from_dict({
        "sentence1": [f"First sentence {i}." for i in range(num_examples)],
        "sentence2": [f"Second sentence {i}." for i in range(num_examples)],
        "label": [i % 2 for i in range(num_examples)],
        "idx": list(range(num_examples)),
    })


class TestLoadMrpc:
    """Tests for load_mrpc()."""

    @patch("src.data.loaders.load_dataset")
    def test_correct_columns(self, mock_load: MagicMock) -> None:
        """Output dataset should have text, label, example_id columns."""
        mock_load.return_value = _make_mock_mrpc()
        ds = load_mrpc()
        assert "text" in ds.column_names
        assert "label" in ds.column_names
        assert "example_id" in ds.column_names
        assert "sentence1" not in ds.column_names
        assert "sentence2" not in ds.column_names

    @patch("src.data.loaders.load_dataset")
    def test_sep_in_text(self, mock_load: MagicMock) -> None:
        """Text should contain the [SEP] separator between sentences."""
        mock_load.return_value = _make_mock_mrpc()
        ds = load_mrpc()
        for text in ds["text"]:
            assert " [SEP] " in text

    @patch("src.data.loaders.load_dataset")
    def test_text_concatenation(self, mock_load: MagicMock) -> None:
        """Text should be sentence1 + ' [SEP] ' + sentence2."""
        mock_load.return_value = _make_mock_mrpc()
        ds = load_mrpc()
        assert ds["text"][0] == "First sentence 0. [SEP] Second sentence 0."

    @patch("src.data.loaders.load_dataset")
    def test_label_range(self, mock_load: MagicMock) -> None:
        """Labels should be 0 or 1."""
        mock_load.return_value = _make_mock_mrpc()
        ds = load_mrpc()
        assert all(label in (0, 1) for label in ds["label"])

    @patch("src.data.loaders.load_dataset")
    def test_subsampling(self, mock_load: MagicMock) -> None:
        """Providing max_examples should reduce dataset size."""
        mock_load.return_value = _make_mock_mrpc(20)
        ds = load_mrpc(max_examples=5)
        assert len(ds) == 5


# ---------------------------------------------------------------------------
# LAMA T-REx loader tests
# ---------------------------------------------------------------------------


def _make_mock_lama_trex(num_examples: int = 12) -> Dataset:
    """Create a mock LAMA T-REx dataset matching the HF schema."""
    predicates = ["P17", "P19", "P36"]
    return Dataset.from_dict({
        "masked_sentences": [
            [f"Subject {i} is located in [MASK] country."]
            for i in range(num_examples)
        ],
        "obj_label": [f"Object_{i}" for i in range(num_examples)],
        "sub_label": [f"Subject_{i}" for i in range(num_examples)],
        "predicate_id": [predicates[i % len(predicates)] for i in range(num_examples)],
        "obj_uri": [f"Q{i}" for i in range(num_examples)],
        "sub_uri": [f"Q{100 + i}" for i in range(num_examples)],
        "uuid": [f"uuid-{i}" for i in range(num_examples)],
    })


class TestLoadLamaTrex:
    """Tests for load_lama_trex()."""

    @patch("src.data.loaders.load_dataset")
    def test_correct_columns(self, mock_load: MagicMock) -> None:
        """Output should have text, label, example_id, and metadata columns."""
        mock_load.return_value = _make_mock_lama_trex()
        ds = load_lama_trex()
        assert "text" in ds.column_names
        assert "label" in ds.column_names
        assert "example_id" in ds.column_names
        assert "obj_label" in ds.column_names
        assert "sub_label" in ds.column_names
        assert "predicate_id" in ds.column_names

    @patch("src.data.loaders.load_dataset")
    def test_integer_labels(self, mock_load: MagicMock) -> None:
        """Labels should be contiguous integers mapping predicate_id."""
        mock_load.return_value = _make_mock_lama_trex()
        ds = load_lama_trex()
        labels = ds["label"]
        assert all(isinstance(l, int) for l in labels)
        # 3 unique predicates → labels in {0, 1, 2}
        assert set(labels) == {0, 1, 2}

    @patch("src.data.loaders.load_dataset")
    def test_text_truncated_at_mask(self, mock_load: MagicMock) -> None:
        """Text should be the masked sentence truncated before [MASK]."""
        mock_load.return_value = _make_mock_lama_trex()
        ds = load_lama_trex()
        for text in ds["text"]:
            assert "[MASK]" not in text
            assert len(text) > 0

    @patch("src.data.loaders.load_dataset")
    def test_metadata_preserved(self, mock_load: MagicMock) -> None:
        """obj_label, sub_label, predicate_id should be preserved."""
        mock_load.return_value = _make_mock_lama_trex()
        ds = load_lama_trex()
        assert ds["obj_label"][0] == "Object_0"
        assert ds["sub_label"][0] == "Subject_0"
        assert ds["predicate_id"][0] in {"P17", "P19", "P36"}

    @patch("src.data.loaders.load_dataset")
    def test_relations_filter(self, mock_load: MagicMock) -> None:
        """Providing relations should keep only matching examples."""
        mock_load.return_value = _make_mock_lama_trex(12)
        ds = load_lama_trex(relations=["P17"])
        predicates = ds["predicate_id"]
        assert all(p == "P17" for p in predicates)

    @patch("src.data.loaders.load_dataset")
    def test_subsampling(self, mock_load: MagicMock) -> None:
        """Providing max_examples should reduce dataset size."""
        mock_load.return_value = _make_mock_lama_trex(20)
        ds = load_lama_trex(max_examples=5)
        assert len(ds) == 5
