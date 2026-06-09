"""
Hidden state extraction module.

Provides utilities for loading Pythia models and extracting intermediate
hidden representations from transformer layers. The primary entry point
is the smoke test, which validates that hidden state extraction works
end-to-end for a given model.
"""

from src.extraction.smoke_test import (
    extract_hidden_states,
    get_last_non_padding_representation,
    get_mean_pooled_representation,
    load_model_and_tokenizer,
    run_smoke_test,
    set_seed,
    validate_hidden_states,
)
