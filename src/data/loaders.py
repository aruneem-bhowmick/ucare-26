"""
Data loaders for standard HuggingFace datasets.

Each loader downloads (or loads from cache) a HuggingFace dataset and
returns a ``datasets.Dataset`` with a standardized schema:

- ``text`` (str): input text for the model
- ``label`` (int): integer class label
- ``example_id`` (int): unique example identifier

Additional metadata columns may be preserved depending on the task.
"""

import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_sst2(
    split: str = "validation",
    max_examples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load the SST-2 binary sentiment classification dataset.

    Source: ``datasets.load_dataset("glue", "sst2")``.

    Renames ``sentence`` to ``text`` and ``idx`` to ``example_id``.
    The ``label`` column (0 = negative, 1 = positive) is kept as-is.

    Args:
        split: Dataset split to load (e.g. ``"train"``,
            ``"validation"``).
        max_examples: If provided, subsample to this many examples.
        seed: Random seed for reproducible subsampling.

    Returns:
        A ``Dataset`` with columns ``text``, ``label``, ``example_id``.
    """
    ds = load_dataset("glue", "sst2", split=split)

    ds = ds.rename_column("sentence", "text")
    ds = ds.rename_column("idx", "example_id")

    if max_examples is not None and max_examples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_examples))

    logger.info("Loaded SST-2 (%s): %d examples", split, len(ds))
    return ds


def load_mrpc(
    split: str = "validation",
    max_examples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load the MRPC paraphrase detection dataset.

    Source: ``datasets.load_dataset("glue", "mrpc")``.

    Concatenates ``sentence1`` and ``sentence2`` with a ``" [SEP] "``
    separator into a single ``text`` column. Renames ``idx`` to
    ``example_id``. The ``label`` column (0 = not paraphrase,
    1 = paraphrase) is kept as-is.

    Args:
        split: Dataset split to load (e.g. ``"train"``,
            ``"validation"``).
        max_examples: If provided, subsample to this many examples.
        seed: Random seed for reproducible subsampling.

    Returns:
        A ``Dataset`` with columns ``text``, ``label``, ``example_id``.
    """
    ds = load_dataset("glue", "mrpc", split=split)

    ds = ds.map(
        lambda ex: {"text": ex["sentence1"] + " [SEP] " + ex["sentence2"]},
        remove_columns=["sentence1", "sentence2"],
    )
    ds = ds.rename_column("idx", "example_id")

    if max_examples is not None and max_examples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_examples))

    logger.info("Loaded MRPC (%s): %d examples", split, len(ds))
    return ds


def load_lama_trex(
    split: str = "train",
    max_examples: int | None = None,
    seed: int = 42,
    relations: list[str] | None = None,
) -> Dataset:
    """Load the LAMA T-REx factual knowledge probing dataset.

    Source: ``datasets.load_dataset("lama", "trex")``.

    For each example the ``masked_sentence`` is truncated at the
    ``[MASK]`` token to form a prompt. The ``predicate_id`` strings
    are mapped to contiguous integer labels for relation-type
    classification.

    Args:
        split: Dataset split to load.  T-REx only has ``"train"``.
        max_examples: If provided, subsample to this many examples.
        seed: Random seed for reproducible subsampling.
        relations: If provided, keep only examples whose
            ``predicate_id`` is in this list.

    Returns:
        A ``Dataset`` with columns ``text``, ``label``,
        ``example_id``, ``obj_label``, ``sub_label``,
        ``predicate_id``.
    """
    ds = load_dataset("lama", "trex", split=split)

    if relations is not None:
        ds = ds.filter(lambda ex: ex["predicate_id"] in set(relations))

    # Build a mapping from predicate_id strings to contiguous ints.
    unique_predicates = sorted(set(ds["predicate_id"]))
    predicate_to_idx: dict[str, int] = {
        pred: idx for idx, pred in enumerate(unique_predicates)
    }

    def _transform(example: dict, idx: int) -> dict:
        masked = example["masked_sentences"][0]
        # Truncate at [MASK] to form the prompt.
        mask_pos = masked.find("[MASK]")
        if mask_pos >= 0:
            text = masked[:mask_pos].rstrip()
        else:
            text = masked

        return {
            "text": text,
            "label": predicate_to_idx[example["predicate_id"]],
            "example_id": idx,
            "obj_label": example["obj_label"],
            "sub_label": example["sub_label"],
            "predicate_id": example["predicate_id"],
        }

    columns_to_remove = [
        c for c in ds.column_names
        if c not in {"obj_label", "sub_label", "predicate_id"}
    ]
    ds = ds.map(
        _transform,
        with_indices=True,
        remove_columns=columns_to_remove,
    )

    if max_examples is not None and max_examples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_examples))

    logger.info("Loaded LAMA T-REx (%s): %d examples, %d relations",
                split, len(ds), len(unique_predicates))
    return ds
