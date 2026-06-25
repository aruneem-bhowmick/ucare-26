"""
Hook-based hidden state capture from transformer blocks.

Provides a context manager for registering PyTorch forward hooks on
GPTNeoX transformer layers to capture intermediate hidden states
without relying on ``output_hidden_states=True``. This approach gives
finer-grained control over which layers are tapped and avoids
materialising all hidden states when only a subset is needed.
"""

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class HookManager:
    """Context manager for registering forward hooks on GPTNeoX blocks.

    Registers hooks on the embedding layer and selected transformer
    blocks to capture their output tensors during a forward pass.
    The captured tensors are stored in order (embedding first, then
    layers) and can be retrieved via ``get_hidden_states()``.

    Args:
        model: A HuggingFace ``GPTNeoXForCausalLM`` (or compatible)
            model instance.
        layer_indices: List of transformer block indices to hook.
            ``None`` means all layers. Indices are zero-based and
            refer to positions in ``model.gpt_neox.layers``.
        tap_point: Which part of the block to capture. Currently
            only ``"residual"`` (post-block output) is supported.

    Example::

        with HookManager(model, layer_indices=[0, 5]) as hm:
            model(**inputs)
            states = hm.get_hidden_states()
            # states is (embedding_output, layer_0_output, layer_5_output)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int] | None = None,
        tap_point: str = "residual",
    ) -> None:
        self._model = model
        self._tap_point = tap_point
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._captured: list[torch.Tensor] = []

        # Resolve the GPTNeoX backbone
        if not hasattr(model, "gpt_neox"):
            raise ValueError(
                "Model does not have a 'gpt_neox' attribute. "
                "HookManager currently only supports GPTNeoX models."
            )

        num_layers = len(model.gpt_neox.layers)

        if layer_indices is None:
            self._layer_indices = list(range(num_layers))
        else:
            for idx in layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"Layer index {idx} out of range for model "
                        f"with {num_layers} layers (valid: 0-{num_layers - 1})"
                    )
            self._layer_indices = sorted(layer_indices)

    def __enter__(self) -> "HookManager":
        """Register forward hooks on the embedding layer and selected blocks."""
        self._captured = []
        self._handles = []

        # Hook the embedding layer (index 0 in hidden_states convention)
        embed_layer = self._model.gpt_neox.embed_in

        def _embed_hook(
            module: nn.Module, input: Any, output: torch.Tensor
        ) -> None:
            self._captured.append(output.detach())

        handle = embed_layer.register_forward_hook(_embed_hook)
        self._handles.append(handle)

        # Hook each selected transformer block
        for idx in self._layer_indices:
            layer = self._model.gpt_neox.layers[idx]

            def _layer_hook(
                module: nn.Module, input: Any, output: Any, _idx: int = idx
            ) -> None:
                # GPTNeoXLayer forward returns a tuple where output[0]
                # is the residual stream tensor of shape (batch, seq, hidden)
                if isinstance(output, tuple):
                    self._captured.append(output[0].detach())
                else:
                    self._captured.append(output.detach())

            handle = layer.register_forward_hook(_layer_hook)
            self._handles.append(handle)

        logger.debug(
            "Registered %d hooks (1 embedding + %d layers)",
            len(self._handles),
            len(self._layer_indices),
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        logger.debug("All hooks removed")

    def get_hidden_states(self) -> tuple[torch.Tensor, ...]:
        """Return captured hidden states as a tuple.

        The tuple contains the embedding output at index 0 followed by
        the outputs of the selected transformer blocks, matching the
        convention of ``model(..., output_hidden_states=True).hidden_states``.

        Returns:
            Tuple of tensors, each of shape ``(batch, seq, hidden)``.
        """
        return tuple(self._captured)

    def clear(self) -> None:
        """Reset captured state for the next batch."""
        self._captured = []
