"""
Tests for synthetic data generators.

Validates correctness, label distribution, seed determinism, and
output schema for Dyck-k, modular arithmetic, and periodic table
generators.
"""

import re

import pytest
from datasets import Dataset

from src.data.synthetic import (
    _is_wellformed_dyck,
    generate_dyck,
    generate_modular_arithmetic,
    generate_periodic_table,
)


# ---------------------------------------------------------------------------
# Dyck generator tests
# ---------------------------------------------------------------------------


class TestGenerateDyck:
    """Tests for generate_dyck()."""

    def test_correct_columns(self) -> None:
        """Output should have text, label, example_id columns."""
        ds = generate_dyck(k=2, num_examples=20)
        assert set(ds.column_names) == {"text", "label", "example_id"}

    def test_correct_size(self) -> None:
        """Output should have the requested number of examples."""
        ds = generate_dyck(k=2, num_examples=50)
        assert len(ds) == 50

    def test_label_distribution(self) -> None:
        """Labels should be roughly 50/50 positive and negative."""
        ds = generate_dyck(k=2, num_examples=100)
        labels = ds["label"]
        num_positive = sum(1 for l in labels if l == 1)
        num_negative = sum(1 for l in labels if l == 0)
        assert num_positive == 50
        assert num_negative == 50

    def test_wellformed_sequences_validate(self) -> None:
        """Sequences with label=1 should pass the well-formedness check."""
        ds = generate_dyck(k=3, num_examples=100, seed=99)
        for text, label in zip(ds["text"], ds["label"]):
            if label == 1:
                assert _is_wellformed_dyck(text), (
                    f"Well-formed sequence failed validation: {text!r}"
                )

    def test_malformed_sequences_fail(self) -> None:
        """Sequences with label=0 should fail the well-formedness check."""
        ds = generate_dyck(k=3, num_examples=100, seed=99)
        for text, label in zip(ds["text"], ds["label"]):
            if label == 0:
                assert not _is_wellformed_dyck(text), (
                    f"Malformed sequence passed validation: {text!r}"
                )

    def test_seed_determinism(self) -> None:
        """Same seed should produce identical datasets."""
        ds1 = generate_dyck(k=2, num_examples=30, seed=123)
        ds2 = generate_dyck(k=2, num_examples=30, seed=123)
        assert ds1["text"] == ds2["text"]
        assert ds1["label"] == ds2["label"]

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different datasets."""
        ds1 = generate_dyck(k=2, num_examples=30, seed=1)
        ds2 = generate_dyck(k=2, num_examples=30, seed=2)
        assert ds1["text"] != ds2["text"]

    def test_invalid_k_raises(self) -> None:
        """k outside [1, 4] should raise ValueError."""
        with pytest.raises(ValueError, match="k must be between"):
            generate_dyck(k=0)
        with pytest.raises(ValueError, match="k must be between"):
            generate_dyck(k=5)

    def test_bracket_types_limited_by_k(self) -> None:
        """With k=1, only parentheses should appear."""
        ds = generate_dyck(k=1, num_examples=50, seed=42)
        for text in ds["text"]:
            # Only ( and ) should appear, no other bracket types
            allowed = set("()")
            brackets_in_text = set(ch for ch in text if not ch.isalnum())
            assert brackets_in_text <= allowed, (
                f"k=1 sequence contains disallowed brackets: {text!r}"
            )

    def test_example_ids_sequential(self) -> None:
        """Example IDs should be sequential starting from 0."""
        ds = generate_dyck(k=2, num_examples=20)
        assert ds["example_id"] == list(range(20))


# ---------------------------------------------------------------------------
# Modular arithmetic generator tests
# ---------------------------------------------------------------------------


class TestGenerateModularArithmetic:
    """Tests for generate_modular_arithmetic()."""

    def test_correct_columns(self) -> None:
        """Output should have text, label, example_id columns."""
        ds = generate_modular_arithmetic(p=5, num_examples=20)
        assert set(ds.column_names) == {"text", "label", "example_id"}

    def test_correct_size(self) -> None:
        """Output should have the requested number of examples."""
        ds = generate_modular_arithmetic(p=7, num_examples=50)
        assert len(ds) == 50

    def test_labels_in_range(self) -> None:
        """Labels should be in [0, p)."""
        p = 7
        ds = generate_modular_arithmetic(p=p, num_examples=100)
        assert all(0 <= label < p for label in ds["label"])

    def test_computed_label_correct(self) -> None:
        """Each label should equal (a op b) mod p."""
        p = 11
        ds = generate_modular_arithmetic(p=p, num_examples=100, seed=42)
        for text, label in zip(ds["text"], ds["label"]):
            # Parse "( a op b ) mod p ="
            match = re.match(r"\( (\d+) ([+\-*]) (\d+) \) mod (\d+) =", text)
            assert match is not None, f"Could not parse: {text!r}"
            a, op, b, mod = int(match[1]), match[2], int(match[3]), int(match[4])
            assert mod == p

            if op == "+":
                expected = (a + b) % p
            elif op == "-":
                expected = (a - b) % p
            else:
                expected = (a * b) % p

            assert label == expected, (
                f"For {text!r}: expected {expected}, got {label}"
            )

    def test_seed_determinism(self) -> None:
        """Same seed should produce identical datasets."""
        ds1 = generate_modular_arithmetic(p=5, num_examples=30, seed=42)
        ds2 = generate_modular_arithmetic(p=5, num_examples=30, seed=42)
        assert ds1["text"] == ds2["text"]
        assert ds1["label"] == ds2["label"]

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different datasets."""
        ds1 = generate_modular_arithmetic(p=7, num_examples=30, seed=1)
        ds2 = generate_modular_arithmetic(p=7, num_examples=30, seed=2)
        assert ds1["text"] != ds2["text"]

    def test_operations_subset(self) -> None:
        """Restricting operations should only produce those operations."""
        ds = generate_modular_arithmetic(
            p=5, num_examples=50, operations=["+"], seed=42
        )
        for text in ds["text"]:
            assert "+" in text
            assert "*" not in text
            # Note: '-' can appear in negative numbers context so check op position
            match = re.match(r"\( \d+ ([+\-*]) \d+ \)", text)
            assert match is not None
            assert match[1] == "+"

    def test_invalid_operation_raises(self) -> None:
        """Invalid operations should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid operations"):
            generate_modular_arithmetic(operations=["/"])

    def test_example_ids_sequential(self) -> None:
        """Example IDs should be sequential starting from 0."""
        ds = generate_modular_arithmetic(p=5, num_examples=20)
        assert ds["example_id"] == list(range(20))


# ---------------------------------------------------------------------------
# Periodic table generator tests
# ---------------------------------------------------------------------------


class TestGeneratePeriodicTable:
    """Tests for generate_periodic_table()."""

    def test_correct_columns(self) -> None:
        """Output should have all expected columns."""
        ds = generate_periodic_table(num_examples=20)
        expected = {
            "text", "label", "example_id",
            "element_name", "symbol",
            "atomic_number", "group", "period",
        }
        assert set(ds.column_names) == expected

    def test_correct_size(self) -> None:
        """Output should have the requested number of examples."""
        ds = generate_periodic_table(num_examples=50)
        assert len(ds) == 50

    def test_label_is_atomic_number_by_default(self) -> None:
        """Default label_column='atomic_number' should use atomic numbers."""
        ds = generate_periodic_table(num_examples=50)
        for label, atomic_number in zip(ds["label"], ds["atomic_number"]):
            assert label == atomic_number

    def test_label_column_group(self) -> None:
        """label_column='group' should use group numbers as labels."""
        ds = generate_periodic_table(num_examples=50, label_column="group")
        for label, group in zip(ds["label"], ds["group"]):
            assert label == group

    def test_label_column_period(self) -> None:
        """label_column='period' should use period numbers as labels."""
        ds = generate_periodic_table(num_examples=50, label_column="period")
        for label, period in zip(ds["label"], ds["period"]):
            assert label == period

    def test_invalid_label_column_raises(self) -> None:
        """Invalid label_column should raise ValueError."""
        with pytest.raises(ValueError, match="label_column"):
            generate_periodic_table(label_column="mass")

    def test_atomic_number_range(self) -> None:
        """Atomic numbers should be in [1, 50]."""
        ds = generate_periodic_table(num_examples=100)
        assert all(1 <= an <= 50 for an in ds["atomic_number"])

    def test_element_coverage(self) -> None:
        """With enough examples, multiple distinct elements should appear."""
        ds = generate_periodic_table(num_examples=100, seed=42)
        unique_elements = set(ds["element_name"])
        # 100 examples from 50 elements × 8 templates = 400 total
        # should cover many distinct elements
        assert len(unique_elements) >= 10

    def test_template_variety(self) -> None:
        """Multiple distinct prompt templates should appear in the text."""
        ds = generate_periodic_table(num_examples=200, seed=42)
        texts = ds["text"]
        # Check that we see different template patterns
        has_atomic_number_template = any("has atomic number" in t for t in texts)
        has_group_template = any("is in group" in t for t in texts)
        has_symbol_template = any("symbol" in t.lower() for t in texts)
        assert has_atomic_number_template
        assert has_group_template
        assert has_symbol_template

    def test_seed_determinism(self) -> None:
        """Same seed should produce identical datasets."""
        ds1 = generate_periodic_table(num_examples=50, seed=42)
        ds2 = generate_periodic_table(num_examples=50, seed=42)
        assert ds1["text"] == ds2["text"]
        assert ds1["label"] == ds2["label"]

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different datasets."""
        ds1 = generate_periodic_table(num_examples=50, seed=1)
        ds2 = generate_periodic_table(num_examples=50, seed=2)
        assert ds1["text"] != ds2["text"]

    def test_example_ids_sequential(self) -> None:
        """Example IDs should be sequential starting from 0."""
        ds = generate_periodic_table(num_examples=30)
        assert ds["example_id"] == list(range(30))

    def test_group_range(self) -> None:
        """Groups should be in [1, 18]."""
        ds = generate_periodic_table(num_examples=200)
        assert all(1 <= g <= 18 for g in ds["group"])

    def test_period_range(self) -> None:
        """Periods should be in [1, 5] (elements H through Sn)."""
        ds = generate_periodic_table(num_examples=200)
        assert all(1 <= p <= 5 for p in ds["period"])
