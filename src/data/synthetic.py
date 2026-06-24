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


# -----------------------------------------------------------------------
# Periodic table data
# -----------------------------------------------------------------------

# Built-in data for 50 elements (H through Sn).
# Each tuple: (atomic_number, symbol, name, group, period)
_ELEMENTS: list[tuple[int, str, str, int, int]] = [
    (1, "H", "Hydrogen", 1, 1),
    (2, "He", "Helium", 18, 1),
    (3, "Li", "Lithium", 1, 2),
    (4, "Be", "Beryllium", 2, 2),
    (5, "B", "Boron", 13, 2),
    (6, "C", "Carbon", 14, 2),
    (7, "N", "Nitrogen", 15, 2),
    (8, "O", "Oxygen", 16, 2),
    (9, "F", "Fluorine", 17, 2),
    (10, "Ne", "Neon", 18, 2),
    (11, "Na", "Sodium", 1, 3),
    (12, "Mg", "Magnesium", 2, 3),
    (13, "Al", "Aluminium", 13, 3),
    (14, "Si", "Silicon", 14, 3),
    (15, "P", "Phosphorus", 15, 3),
    (16, "S", "Sulfur", 16, 3),
    (17, "Cl", "Chlorine", 17, 3),
    (18, "Ar", "Argon", 18, 3),
    (19, "K", "Potassium", 1, 4),
    (20, "Ca", "Calcium", 2, 4),
    (21, "Sc", "Scandium", 3, 4),
    (22, "Ti", "Titanium", 4, 4),
    (23, "V", "Vanadium", 5, 4),
    (24, "Cr", "Chromium", 6, 4),
    (25, "Mn", "Manganese", 7, 4),
    (26, "Fe", "Iron", 8, 4),
    (27, "Co", "Cobalt", 9, 4),
    (28, "Ni", "Nickel", 10, 4),
    (29, "Cu", "Copper", 11, 4),
    (30, "Zn", "Zinc", 12, 4),
    (31, "Ga", "Gallium", 13, 4),
    (32, "Ge", "Germanium", 14, 4),
    (33, "As", "Arsenic", 15, 4),
    (34, "Se", "Selenium", 16, 4),
    (35, "Br", "Bromine", 17, 4),
    (36, "Kr", "Krypton", 18, 4),
    (37, "Rb", "Rubidium", 1, 5),
    (38, "Sr", "Strontium", 2, 5),
    (39, "Y", "Yttrium", 3, 5),
    (40, "Zr", "Zirconium", 4, 5),
    (41, "Nb", "Niobium", 5, 5),
    (42, "Mo", "Molybdenum", 6, 5),
    (43, "Tc", "Technetium", 7, 5),
    (44, "Ru", "Ruthenium", 8, 5),
    (45, "Rh", "Rhodium", 9, 5),
    (46, "Pd", "Palladium", 10, 5),
    (47, "Ag", "Silver", 11, 5),
    (48, "Cd", "Cadmium", 12, 5),
    (49, "In", "Indium", 13, 5),
    (50, "Sn", "Tin", 14, 5),
]

# Template pool for generating periodic table prompts.
_PERIODIC_TEMPLATES: list[str] = [
    "The element {name} has atomic number",
    "The atomic number of {name} is",
    "{name} is in group",
    "{name} is in period",
    "The symbol for {name} is",
    "{name} belongs to group",
    "Element {name} has the symbol",
    "The element with symbol {symbol} has atomic number",
]


def generate_periodic_table(
    num_examples: int = 200,
    seed: int = 42,
    label_column: str = "atomic_number",
) -> Dataset:
    """Generate a periodic table structured-knowledge probing dataset.

    Cycles through 50 elements (H through Sn) crossed with a pool
    of prompt templates, shuffles by seed, then slices to
    ``num_examples``.

    Args:
        num_examples: Number of examples to generate.
        seed: Random seed for reproducibility.
        label_column: Which property to use as the primary label —
            ``"atomic_number"`` (regression), ``"group"``, or
            ``"period"`` (classification).

    Returns:
        A ``Dataset`` with columns ``text``, ``label``,
        ``example_id``, ``element_name``, ``symbol``,
        ``atomic_number``, ``group``, ``period``.
    """
    valid_label_columns = {"atomic_number", "group", "period"}
    if label_column not in valid_label_columns:
        raise ValueError(
            f"label_column must be one of {valid_label_columns}, "
            f"got {label_column!r}"
        )

    # Build all element × template combinations.
    rows: list[dict] = []
    for atomic_number, symbol, name, group, period in _ELEMENTS:
        for template in _PERIODIC_TEMPLATES:
            text = template.format(name=name, symbol=symbol)
            row = {
                "text": text,
                "element_name": name,
                "symbol": symbol,
                "atomic_number": atomic_number,
                "group": group,
                "period": period,
            }
            rows.append(row)

    # Shuffle deterministically then slice.
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:num_examples]

    # Assign labels and example IDs.
    texts = [r["text"] for r in rows]
    labels = [r[label_column] for r in rows]
    example_ids = list(range(len(rows)))

    ds = Dataset.from_dict({
        "text": texts,
        "label": labels,
        "example_id": example_ids,
        "element_name": [r["element_name"] for r in rows],
        "symbol": [r["symbol"] for r in rows],
        "atomic_number": [r["atomic_number"] for r in rows],
        "group": [r["group"] for r in rows],
        "period": [r["period"] for r in rows],
    })

    logger.info("Generated periodic table dataset: %d examples, "
                "label=%s", num_examples, label_column)
    return ds
