"""
Tests for the task specification registry.

Validates YAML config loading, TaskSpec field completeness,
programmatic lookup by name, and error handling for unknown tasks.
"""

import textwrap
from pathlib import Path

import pytest

from src.data.tasks import (
    TaskSpec,
    get_task_spec,
    load_task_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_YAML = textwrap.dedent("""\
    tasks:
      sst2:
        family: "shallow_semantic"
        label_type: "classification"
        num_classes: 2
        extraction_position: "last_token"
        description: "Binary sentiment classification"

      dyck:
        family: "algorithmic"
        label_type: "classification"
        num_classes: 2
        extraction_position: "last_token"
        description: "Balanced bracket detection"

      periodic_table:
        family: "factual_lookup"
        label_type: "regression"
        num_classes: null
        extraction_position: "last_token"
        description: "Element property prediction"
""")


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal task registry YAML to a temp file.

    Returns:
        Path to the temporary YAML file.
    """
    config_file = tmp_path / "tasks.yaml"
    config_file.write_text(SAMPLE_YAML)
    return config_file


# ---------------------------------------------------------------------------
# TestLoadTaskRegistry
# ---------------------------------------------------------------------------


class TestLoadTaskRegistry:
    """Tests for load_task_registry()."""

    def test_loads_all_tasks(self, sample_config: Path) -> None:
        """Registry should contain all tasks defined in the YAML."""
        registry = load_task_registry(sample_config)
        assert set(registry.keys()) == {"sst2", "dyck", "periodic_table"}

    def test_returns_task_spec_instances(self, sample_config: Path) -> None:
        """Every value in the registry should be a TaskSpec."""
        registry = load_task_registry(sample_config)
        for spec in registry.values():
            assert isinstance(spec, TaskSpec)

    def test_per_task_fields(self, sample_config: Path) -> None:
        """Per-task fields should match the YAML entries."""
        registry = load_task_registry(sample_config)
        sst2 = registry["sst2"]
        assert sst2.family == "shallow_semantic"
        assert sst2.label_type == "classification"
        assert sst2.num_classes == 2
        assert sst2.extraction_position == "last_token"
        assert sst2.description == "Binary sentiment classification"

    def test_null_num_classes(self, sample_config: Path) -> None:
        """Tasks with null num_classes should have None."""
        registry = load_task_registry(sample_config)
        pt = registry["periodic_table"]
        assert pt.num_classes is None

    def test_task_spec_is_frozen(self, sample_config: Path) -> None:
        """TaskSpec instances should be immutable."""
        registry = load_task_registry(sample_config)
        spec = registry["sst2"]
        with pytest.raises(AttributeError):
            spec.family = "other"  # type: ignore[misc]

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        """Loading from a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_task_registry(tmp_path / "nonexistent.yaml")

    def test_name_field_matches_dict_key(self, sample_config: Path) -> None:
        """The TaskSpec.name attribute should match its registry dict key."""
        registry = load_task_registry(sample_config)
        for key, spec in registry.items():
            assert spec.name == key


# ---------------------------------------------------------------------------
# TestGetTaskSpec
# ---------------------------------------------------------------------------


class TestGetTaskSpec:
    """Tests for get_task_spec()."""

    def test_valid_name(self, sample_config: Path) -> None:
        """Looking up a valid name should return the correct TaskSpec."""
        spec = get_task_spec("sst2", sample_config)
        assert spec.name == "sst2"
        assert spec.family == "shallow_semantic"

    def test_unknown_name_raises(self, sample_config: Path) -> None:
        """Looking up a nonexistent name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown task name"):
            get_task_spec("nonexistent_task", sample_config)

    def test_error_message_lists_available(self, sample_config: Path) -> None:
        """The KeyError message should list available task names."""
        with pytest.raises(KeyError, match="sst2"):
            get_task_spec("bad-name", sample_config)


# ---------------------------------------------------------------------------
# TestTaskSpecFields
# ---------------------------------------------------------------------------


class TestTaskSpecFields:
    """Tests for TaskSpec field types and constraints."""

    def test_all_required_fields_present(self, sample_config: Path) -> None:
        """Every required field should be populated (not None except num_classes)."""
        spec = get_task_spec("sst2", sample_config)
        for field_name in TaskSpec.__dataclass_fields__:
            if field_name == "num_classes":
                continue
            assert getattr(spec, field_name) is not None, (
                f"Field {field_name!r} is None"
            )

    def test_family_values(self, sample_config: Path) -> None:
        """Family should be one of the allowed values."""
        allowed = {"shallow_semantic", "factual_lookup", "algorithmic"}
        registry = load_task_registry(sample_config)
        for spec in registry.values():
            assert spec.family in allowed, (
                f"{spec.name}: family {spec.family!r} not in {allowed}"
            )

    def test_label_type_values(self, sample_config: Path) -> None:
        """Label type should be classification or regression."""
        allowed = {"classification", "regression"}
        registry = load_task_registry(sample_config)
        for spec in registry.values():
            assert spec.label_type in allowed, (
                f"{spec.name}: label_type {spec.label_type!r} not in {allowed}"
            )

    def test_extraction_position_values(self, sample_config: Path) -> None:
        """Extraction position should be last_token or mean."""
        allowed = {"last_token", "mean"}
        registry = load_task_registry(sample_config)
        for spec in registry.values():
            assert spec.extraction_position in allowed, (
                f"{spec.name}: extraction_position "
                f"{spec.extraction_position!r} not in {allowed}"
            )


# ---------------------------------------------------------------------------
# TestRealConfig
# ---------------------------------------------------------------------------


class TestRealConfig:
    """Tests against the actual configs/tasks.yaml shipped in the repo."""

    def test_real_config_loads(self) -> None:
        """The shipped tasks.yaml should load without errors."""
        registry = load_task_registry()
        assert len(registry) == 6

    def test_real_config_expected_keys(self) -> None:
        """The shipped registry should contain all six task keys."""
        registry = load_task_registry()
        expected = {
            "sst2",
            "mrpc",
            "lama_trex",
            "periodic_table",
            "dyck",
            "modular_arithmetic",
        }
        assert set(registry.keys()) == expected

    def test_real_config_families(self) -> None:
        """Each task should have the expected family assignment."""
        registry = load_task_registry()
        assert registry["sst2"].family == "shallow_semantic"
        assert registry["mrpc"].family == "shallow_semantic"
        assert registry["lama_trex"].family == "factual_lookup"
        assert registry["periodic_table"].family == "factual_lookup"
        assert registry["dyck"].family == "algorithmic"
        assert registry["modular_arithmetic"].family == "algorithmic"

    def test_real_config_label_types(self) -> None:
        """Each task should have the expected label type."""
        registry = load_task_registry()
        for name in ("sst2", "mrpc", "lama_trex", "dyck", "modular_arithmetic"):
            assert registry[name].label_type == "classification"
        assert registry["periodic_table"].label_type == "regression"

    def test_all_tasks_have_descriptions(self) -> None:
        """Every task in the real config should have a non-empty description."""
        registry = load_task_registry()
        for key, spec in registry.items():
            assert spec.description, (
                f"Task {key!r} has empty description"
            )
