"""
Surface-feature baseline probes for Dyck-k.

The shuffled-label control (``selectivity.py``) only catches a probe
that memorizes; it cannot catch a probe that reads a real but shallow
input statistic correlated with the label, because that statistic
survives into the hidden states. These baselines fit the same
cross-validated linear probe directly on hand-built input features,
so a hidden-state probe's score can be compared against what the raw
input already gives away:

- ``"surface"``: sequence length, per-bracket-character counts, and
  one-hot first, second, second-to-last and last bracket. Should sit
  at chance when the corruption preserves length, counts and
  boundaries; the second and second-to-last slots catch near-boundary
  cues such as a closer right after the first bracket, which depth
  violations make more common.
- ``"bigram"``: the surface features plus counts of every adjacent
  bracket pair. Many malformed strings contain a locally impossible
  pair such as ``"( ]"``, so this measures how much of the task is
  solvable from local context without tracking the stack.
"""

import logging
from typing import Any

import numpy as np

from src.data.tasks import TaskSpec
from src.probing.cross_validation import CVResult, run_cross_validated_probe

logger = logging.getLogger(__name__)

# Bracket characters in a fixed order, matching synthetic._BRACKET_PAIRS.
_BRACKETS: list[str] = ["(", ")", "[", "]", "{", "}", "<", ">"]

BASELINE_KINDS: tuple[str, ...] = ("surface", "bigram")


def dyck_features(texts: list[str], kind: str, k: int = 3) -> np.ndarray:
    """Build hand-crafted input features for Dyck-k strings.

    Args:
        texts: Bracket strings, with or without separating spaces.
        kind: ``"surface"`` or ``"bigram"`` (see module docstring).
        k: Number of bracket types in use.

    Returns:
        A float feature matrix of shape ``(len(texts), n_features)``.

    Raises:
        ValueError: If *kind* is not one of ``BASELINE_KINDS``.
    """
    if kind not in BASELINE_KINDS:
        raise ValueError(f"kind must be one of {BASELINE_KINDS}, got {kind!r}")

    alphabet = _BRACKETS[: 2 * k]
    index = {ch: i for i, ch in enumerate(alphabet)}
    a = len(alphabet)

    rows: list[np.ndarray] = []
    for text in texts:
        chars = [ch for ch in text if ch in index]
        counts = np.zeros(a)
        # One-hot bracket at positions 0, 1, -2 and -1.
        edges = np.zeros((4, a))
        for ch in chars:
            counts[index[ch]] += 1
        if len(chars) >= 2:
            for slot, pos in enumerate((0, 1, -2, -1)):
                edges[slot, index[chars[pos]]] = 1
        elif chars:
            edges[0, index[chars[0]]] = edges[3, index[chars[0]]] = 1
        parts = [np.array([len(chars)]), counts, edges.ravel()]
        if kind == "bigram":
            bigrams = np.zeros(a * a)
            for x, y in zip(chars, chars[1:]):
                bigrams[index[x] * a + index[y]] += 1
            parts.append(bigrams)
        rows.append(np.concatenate(parts))
    return np.stack(rows)


def run_dyck_baseline(
    texts: list[str],
    y: Any,
    task_spec: TaskSpec,
    config: dict[str, Any],
    kind: str,
    k: int = 3,
) -> CVResult:
    """Cross-validate a linear probe on Dyck input features.

    Uses ``run_cross_validated_probe`` with the same *config* as the
    hidden-state probes, so folds, regularization sweep and probe type
    match and the scores are directly comparable.

    Args:
        texts: Bracket strings, one per example.
        y: Labels, shape ``(len(texts),)``.
        task_spec: Task specification (classification dispatch).
        config: Parsed probing configuration.
        kind: ``"surface"`` or ``"bigram"``.
        k: Number of bracket types in use.

    Returns:
        The ``CVResult`` for the baseline probe.
    """
    X = dyck_features(texts, kind=kind, k=k)
    result = run_cross_validated_probe(X, y, task_spec, config)
    logger.info(
        "Dyck %s baseline: mean_score=%.4f (%d features)",
        kind,
        result.mean_score,
        X.shape[1],
    )
    return result
