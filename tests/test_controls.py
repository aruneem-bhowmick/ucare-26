"""
Tests for the shuffled-label control generator.

Validates that label permutation preserves the marginal distribution,
breaks the label–feature association, respects seed determinism, and
handles edge cases correctly.
"""

import pytest
from datasets import Dataset

from src.data.controls import generate_shuffled_labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_dataset() -> Dataset:
    """A small classification dataset with known labels."""
    return Dataset.from_dict({
        "text": [f"sentence {i}" for i in range(20)],
        "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "example_id": list(range(20)),
    })


@pytest.fixture
def regression_dataset() -> Dataset:
    """A small regression dataset with float labels."""
    return Dataset.from_dict({
        "text": [f"element {i}" for i in range(10)],
        "label": [1.5, 3.2, 7.0, 12.1, 4.4, 8.8, 2.3, 6.6, 9.9, 0.1],
        "example_id": list(range(10)),
    })


@pytest.fixture
def dataset_with_metadata() -> Dataset:
    """A dataset with extra metadata columns."""
    return Dataset.from_dict({
        "text": ["alpha", "beta", "gamma", "delta"],
        "label": [0, 1, 2, 3],
        "example_id": [0, 1, 2, 3],
        "group": [1, 1, 2, 2],
        "source": ["a", "b", "a", "b"],
    })


# ---------------------------------------------------------------------------
# TestLabelDistribution
# ---------------------------------------------------------------------------


class TestLabelDistribution:
    """The marginal label distribution must be preserved exactly."""

    def test_classification_counts_preserved(
        self, classification_dataset: Dataset
    ) -> None:
        """Class counts should be identical before and after shuffling."""
        original_counts = sorted(classification_dataset["label"])
        control = generate_shuffled_labels(classification_dataset)
        shuffled_counts = sorted(control["label"])
        assert original_counts == shuffled_counts

    def test_regression_values_preserved(
        self, regression_dataset: Dataset
    ) -> None:
        """Regression label values should be the same multiset."""
        original_sorted = sorted(regression_dataset["label"])
        control = generate_shuffled_labels(regression_dataset)
        shuffled_sorted = sorted(control["label"])
        assert original_sorted == shuffled_sorted

    def test_dataset_length_unchanged(
        self, classification_dataset: Dataset
    ) -> None:
        """Shuffled dataset should have the same number of examples."""
        control = generate_shuffled_labels(classification_dataset)
        assert len(control) == len(classification_dataset)


# ---------------------------------------------------------------------------
# TestFeatureDecorrelation
# ---------------------------------------------------------------------------


class TestFeatureDecorrelation:
    """Shuffling should break the label–feature association."""

    def test_labels_differ_from_original(
        self, classification_dataset: Dataset
    ) -> None:
        """Shuffled labels should not be identical to original labels.

        With 20 examples and a 50/50 split, a random permutation
        matching the original exactly is astronomically unlikely.
        """
        control = generate_shuffled_labels(classification_dataset, seed=42)
        assert control["label"] != classification_dataset["label"]

    def test_text_column_unchanged(
        self, classification_dataset: Dataset
    ) -> None:
        """The text column must be untouched by shuffling."""
        control = generate_shuffled_labels(classification_dataset)
        assert control["text"] == classification_dataset["text"]

    def test_example_ids_unchanged(
        self, classification_dataset: Dataset
    ) -> None:
        """Example IDs must be untouched by shuffling."""
        control = generate_shuffled_labels(classification_dataset)
        assert control["example_id"] == classification_dataset["example_id"]


# ---------------------------------------------------------------------------
# TestSeedDeterminism
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    """The permutation must be reproducible given the same seed."""

    def test_same_seed_same_result(
        self, classification_dataset: Dataset
    ) -> None:
        """Two calls with the same seed should produce identical labels."""
        c1 = generate_shuffled_labels(classification_dataset, seed=99)
        c2 = generate_shuffled_labels(classification_dataset, seed=99)
        assert c1["label"] == c2["label"]

    def test_different_seeds_differ(
        self, classification_dataset: Dataset
    ) -> None:
        """Different seeds should produce different permutations."""
        c1 = generate_shuffled_labels(classification_dataset, seed=1)
        c2 = generate_shuffled_labels(classification_dataset, seed=2)
        assert c1["label"] != c2["label"]


# ---------------------------------------------------------------------------
# TestMetadataPreservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    """Non-label columns must pass through untouched."""

    def test_metadata_columns_present(
        self, dataset_with_metadata: Dataset
    ) -> None:
        """All original columns should exist in the output."""
        control = generate_shuffled_labels(dataset_with_metadata)
        for col in dataset_with_metadata.column_names:
            assert col in control.column_names

    def test_metadata_values_unchanged(
        self, dataset_with_metadata: Dataset
    ) -> None:
        """Non-label columns should have identical values."""
        control = generate_shuffled_labels(dataset_with_metadata)
        assert control["text"] == dataset_with_metadata["text"]
        assert control["example_id"] == dataset_with_metadata["example_id"]
        assert control["group"] == dataset_with_metadata["group"]
        assert control["source"] == dataset_with_metadata["source"]

    def test_no_extra_columns(
        self, dataset_with_metadata: Dataset
    ) -> None:
        """No new columns should be introduced."""
        control = generate_shuffled_labels(dataset_with_metadata)
        assert set(control.column_names) == set(
            dataset_with_metadata.column_names
        )


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_missing_label_column_raises(self) -> None:
        """Requesting a non-existent column should raise ValueError."""
        ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        with pytest.raises(ValueError, match="not found"):
            generate_shuffled_labels(ds, label_column="missing")

    def test_single_example(self) -> None:
        """A single-example dataset should return the same label."""
        ds = Dataset.from_dict({
            "text": ["only one"],
            "label": [42],
            "example_id": [0],
        })
        control = generate_shuffled_labels(ds)
        assert control["label"] == [42]

    def test_custom_label_column(self) -> None:
        """Shuffling a non-default label column should work."""
        ds = Dataset.from_dict({
            "text": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "group": [10, 20, 30, 40],
            "example_id": [0, 1, 2, 3],
        })
        control = generate_shuffled_labels(ds, label_column="group")
        # group values permuted, label untouched
        assert sorted(control["group"]) == [10, 20, 30, 40]
        assert control["label"] == ds["label"]

    def test_original_dataset_unmodified(
        self, classification_dataset: Dataset
    ) -> None:
        """The original dataset should not be mutated."""
        original_labels = list(classification_dataset["label"])
        generate_shuffled_labels(classification_dataset, seed=42)
        assert list(classification_dataset["label"]) == original_labels
