"""
Linear probing module.

Provides linear probe classifiers for evaluating the linear
decodability of task-relevant information across transformer layers.
"""

from src.probing.probes import LinearProbe, load_probing_config

__all__ = [
    "LinearProbe",
    "load_probing_config",
]
