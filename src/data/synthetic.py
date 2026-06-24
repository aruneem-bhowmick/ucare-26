"""
Synthetic data generators for algorithmic and structured-knowledge tasks.

Each generator produces a ``datasets.Dataset`` with the standardized
schema (``text``, ``label``, ``example_id``) so that downstream
extraction and probing code can treat every task uniformly.
"""

import logging
import random

from datasets import Dataset

logger = logging.getLogger(__name__)

# Bracket pairs used by the Dyck-k generator, indexed 0..k-1.
_BRACKET_PAIRS: list[tuple[str, str]] = [
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
]


def _generate_wellformed_dyck(
    k: int,
    max_depth: int,
    rng: random.Random,
) -> str:
    """Recursively generate a well-formed Dyck-k string.

    The generation proceeds by choosing at each step whether to nest
    deeper (if depth allows) or to close the current bracket. This
    produces varied sequence lengths and nesting patterns.

    Args:
        k: Number of bracket types (1..4).
        max_depth: Maximum nesting depth.
        rng: Random number generator instance.

    Returns:
        A well-formed Dyck-k string.
    """
    def _expand(depth: int) -> str:
        # Decide how many concatenated bracket groups to generate (1-3).
        num_groups = rng.randint(1, min(3, max(1, max_depth - depth + 1)))
        parts: list[str] = []
        for _ in range(num_groups):
            pair_idx = rng.randint(0, k - 1)
            open_b, close_b = _BRACKET_PAIRS[pair_idx]
            if depth < max_depth and rng.random() < 0.6:
                inner = _expand(depth + 1)
            else:
                inner = ""
            parts.append(open_b + inner + close_b)
        return "".join(parts)

    return _expand(0)


def _corrupt_dyck(sequence: str, rng: random.Random) -> str:
    """Corrupt a well-formed Dyck sequence to make it malformed.

    Applies one of three mutations: swap two characters, delete one
    bracket, or insert a random bracket at a random position.

    Args:
        sequence: A well-formed Dyck string.
        rng: Random number generator instance.

    Returns:
        A corrupted (malformed) Dyck string.
    """
    chars = list(sequence)
    if len(chars) < 2:
        # Too short to corrupt meaningfully; just flip the bracket.
        all_brackets = [b for pair in _BRACKET_PAIRS for b in pair]
        return rng.choice(all_brackets)

    mutation = rng.choice(["swap", "delete", "insert"])

    if mutation == "swap":
        i, j = rng.sample(range(len(chars)), 2)
        chars[i], chars[j] = chars[j], chars[i]
    elif mutation == "delete":
        del chars[rng.randint(0, len(chars) - 1)]
    else:  # insert
        all_brackets = [b for pair in _BRACKET_PAIRS for b in pair]
        pos = rng.randint(0, len(chars))
        chars.insert(pos, rng.choice(all_brackets))

    result = "".join(chars)
    # If the corruption accidentally produced a valid sequence, force
    # an unmatched bracket by appending one.
    if _is_wellformed_dyck(result):
        result += rng.choice(["(", "[", "{", "<"])
    return result


def _is_wellformed_dyck(sequence: str) -> bool:
    """Check whether a bracket sequence is well-formed.

    Args:
        sequence: A string of bracket characters.

    Returns:
        ``True`` if the sequence is well-formed, ``False`` otherwise.
    """
    close_to_open = {close: open_ for open_, close in _BRACKET_PAIRS}
    stack: list[str] = []
    for ch in sequence:
        if ch in {open_ for open_, _ in _BRACKET_PAIRS}:
            stack.append(ch)
        elif ch in close_to_open:
            if not stack or stack[-1] != close_to_open[ch]:
                return False
            stack.pop()
        # Ignore non-bracket characters.
    return len(stack) == 0


def generate_dyck(
    k: int = 3,
    num_examples: int = 1000,
    max_depth: int = 5,
    seed: int = 42,
) -> Dataset:
    """Generate a Dyck-k balanced-bracket dataset.

    Produces a 50/50 split of well-formed (label=1) and malformed
    (label=0) bracket sequences using ``k`` bracket types.

    Args:
        k: Number of bracket types (1..4).
        num_examples: Total number of examples to generate.
        max_depth: Maximum nesting depth for well-formed sequences.
        seed: Random seed for reproducibility.

    Returns:
        A ``Dataset`` with columns ``text``, ``label``, ``example_id``.
    """
    if k < 1 or k > len(_BRACKET_PAIRS):
        raise ValueError(
            f"k must be between 1 and {len(_BRACKET_PAIRS)}, got {k}"
        )

    rng = random.Random(seed)
    num_positive = num_examples // 2
    num_negative = num_examples - num_positive

    texts: list[str] = []
    labels: list[int] = []

    # Well-formed examples (label=1).
    for _ in range(num_positive):
        seq = _generate_wellformed_dyck(k, max_depth, rng)
        texts.append(seq)
        labels.append(1)

    # Malformed examples (label=0).
    for _ in range(num_negative):
        wellformed = _generate_wellformed_dyck(k, max_depth, rng)
        corrupted = _corrupt_dyck(wellformed, rng)
        texts.append(corrupted)
        labels.append(0)

    # Shuffle so positive and negative are interleaved.
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)  # type: ignore[assignment]

    ds = Dataset.from_dict({
        "text": list(texts),
        "label": list(labels),
        "example_id": list(range(num_examples)),
    })

    logger.info("Generated Dyck-%d dataset: %d examples", k, num_examples)
    return ds


def generate_modular_arithmetic(
    p: int = 7,
    num_examples: int = 1000,
    operations: list[str] | None = None,
    seed: int = 42,
) -> Dataset:
    """Generate a modular arithmetic dataset.

    Each example is an expression ``"( a op b ) mod p ="`` with the
    label being ``(a op b) mod p`` as an integer class in ``[0, p)``.

    Args:
        p: The modulus (prime recommended for uniform label
            distribution).
        num_examples: Number of examples to generate.
        operations: Subset of ``["+", "-", "*"]`` to use. Defaults
            to all three.
        seed: Random seed for reproducibility.

    Returns:
        A ``Dataset`` with columns ``text``, ``label``, ``example_id``.
    """
    if operations is None:
        operations = ["+", "-", "*"]

    valid_ops = {"+", "-", "*"}
    invalid = set(operations) - valid_ops
    if invalid:
        raise ValueError(
            f"Invalid operations: {invalid}. Must be subset of {valid_ops}"
        )

    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []

    for _ in range(num_examples):
        a = rng.randint(0, p - 1)
        b = rng.randint(0, p - 1)
        op = rng.choice(operations)

        text = f"( {a} {op} {b} ) mod {p} ="
        if op == "+":
            result = (a + b) % p
        elif op == "-":
            result = (a - b) % p
        else:  # "*"
            result = (a * b) % p

        texts.append(text)
        labels.append(result)

    ds = Dataset.from_dict({
        "text": texts,
        "label": labels,
        "example_id": list(range(num_examples)),
    })

    logger.info("Generated modular arithmetic (mod %d): %d examples",
                p, num_examples)
    return ds
