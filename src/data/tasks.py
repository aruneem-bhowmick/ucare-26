"""
Task specification registry for early-halting experiments.

Provides a typed interface for looking up task metadata from the
YAML registry, mirroring the ``ModelSpec`` / ``load_model_registry``
pattern used in ``src/models.py``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to the default task registry config relative to the project root.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "tasks.yaml"


@dataclass(frozen=True)
class TaskSpec:
    """Immutable specification for a single probing task.

    Attributes:
        name: Short identifier used in configs and output paths
            (e.g. ``"sst2"``).
        family: Task family — one of ``"shallow_semantic"``,
            ``"factual_lookup"``, or ``"algorithmic"``.
        label_type: Type of supervision — ``"classification"`` or
            ``"regression"``.
        num_classes: Number of label classes for classification tasks.
            ``None`` for regression or variable-class tasks.
        extraction_position: Pooling strategy for hidden-state
            extraction — ``"last_token"`` or ``"mean"``.
        description: One-line human-readable description of the task.
    """

    name: str
    family: str
    label_type: str
    num_classes: int | None
    extraction_position: str
    description: str


def load_task_registry(
    config_path: str | Path | None = None,
) -> dict[str, TaskSpec]:
    """Load the full task registry from the YAML configuration file.

    Parses ``configs/tasks.yaml`` (or the path provided) and returns
    a dictionary mapping task keys to their ``TaskSpec`` dataclass
    instances.

    Args:
        config_path: Path to the YAML registry file. Defaults to
            ``configs/tasks.yaml`` relative to the project root.

    Returns:
        A dictionary mapping task keys (e.g. ``"sst2"``) to their
        corresponding ``TaskSpec`` instances.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If required fields are missing from a task entry.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    tasks_raw: dict[str, dict[str, Any]] = raw["tasks"]

    registry: dict[str, TaskSpec] = {}
    for key, entry in tasks_raw.items():
        spec = TaskSpec(
            name=key,
            family=entry["family"],
            label_type=entry["label_type"],
            num_classes=entry.get("num_classes"),
            extraction_position=entry["extraction_position"],
            description=entry["description"],
        )
        registry[key] = spec

    logger.info("Loaded task registry with %d entries", len(registry))
    return registry


def get_task_spec(
    name: str,
    config_path: str | Path | None = None,
) -> TaskSpec:
    """Look up a single task specification by its short name.

    Convenience wrapper around ``load_task_registry`` that returns
    the ``TaskSpec`` for the given name or raises a clear error if
    the name is not found.

    Args:
        name: Short task identifier (e.g. ``"sst2"``).
        config_path: Path to the YAML registry file. Defaults to
            ``configs/tasks.yaml`` relative to the project root.

    Returns:
        The ``TaskSpec`` for the requested task.

    Raises:
        KeyError: If *name* is not present in the registry.
    """
    registry = load_task_registry(config_path)
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(
            f"Unknown task name {name!r}. Available tasks: {available}"
        )
    return registry[name]
