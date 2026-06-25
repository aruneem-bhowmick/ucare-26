"""
Hidden state extraction module.

Provides utilities for loading Pythia models and extracting intermediate
hidden representations from transformer layers. Includes hook-based
capture, standalone pooling strategies, fp16 caching with safetensors
and JSONL manifests, and an end-to-end extraction pipeline orchestrator.
"""

from src.extraction.cache import (
    load_representations,
    save_representations,
    verify_manifest,
)
from src.extraction.hooks import HookManager
from src.extraction.pipeline import ExtractionPipeline
from src.extraction.pooling import last_token_pool, mean_pool, pool_hidden_states
from src.extraction.smoke_test import (
    extract_hidden_states,
    get_last_non_padding_representation,
    get_mean_pooled_representation,
    load_model_and_tokenizer,
    run_pipeline_validation,
    run_smoke_test,
    set_seed,
    validate_hidden_states,
)
