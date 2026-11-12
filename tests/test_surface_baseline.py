"""
Tests for the Dyck surface-feature baseline probes.
"""

import numpy as np
import pytest

from src.data.synthetic import generate_dyck
from src.data.tasks import get_task_spec
from src.probing.probes import load_probing_config
from src.probing.surface_baseline import dyck_features, run_dyck_baseline


class TestDyckFeatures:
    """Tests for dyck_features()."""

    def test_surface_feature_values(self) -> None:
        """Length, per-character counts, and one-hot bracket at
        positions 0, 1, -2 and -1."""
        X = dyck_features(["( [ ] )"], kind="surface", k=2)
        # length, counts (()[]), then one-hots for (, [, ], )
        expected = [
            4, 1, 1, 1, 1,
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
            0, 1, 0, 0,
        ]
        assert X.tolist() == [expected]

    def test_bigram_features_extend_surface(self) -> None:
        """Bigram features append (2k)^2 adjacent-pair counts."""
        surface = dyck_features(["( ] [ )"], kind="surface", k=2)
        bigram = dyck_features(["( ] [ )"], kind="bigram", k=2)
        assert bigram.shape[1] == surface.shape[1] + 16
        np.testing.assert_array_equal(bigram[:, : surface.shape[1]], surface)
        assert bigram[0, surface.shape[1]:].sum() == 3

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            dyck_features(["()"], kind="trigram")


class TestRunDyckBaseline:
    """The surface baseline should be uninformative on generate_dyck data."""

    def test_surface_baseline_near_chance(self) -> None:
        ds = generate_dyck(k=3, num_examples=600, seed=42)
        config = load_probing_config()
        result = run_dyck_baseline(
            list(ds["text"]), np.asarray(ds["label"]),
            get_task_spec("dyck"), config, kind="surface",
        )
        assert result.mean_score < 0.6
