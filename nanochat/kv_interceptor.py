# nanochat/kv_interceptor.py
"""
KVInterceptor: Hook-based capture of K/V pairs pre-RoPE.

Model-specific. Currently supports Qwen architecture.
Registers forward hooks on attention layers to intercept K/V
before rotary position embeddings are applied.
"""
import torch
import torch.nn as nn


class KVInterceptor:
    """Captures K/V projections pre-RoPE from specified layers.

    Uses forward hooks on the attention module's k_proj and v_proj
    to capture K/V before RoPE rotation is applied.
    """
    def __init__(self, model: nn.Module, layer_indices: list,
                 model_type: str = "qwen", capture_hidden_states: bool = False):
        self.layer_indices = layer_indices
        self.model_type = model_type
        self.capture_hidden_states = capture_hidden_states
        self._buffer: dict = {}
        self._hs_buffer: dict = {}  # hidden states per layer
        self._hooks: list = []
        self._register_hooks(model)

    def _get_layer(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get the transformer layer. Qwen: model.model.layers[i]"""
        return model.model.layers[layer_idx]

    def _get_attention_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get the attention module for a layer. Qwen: model.model.layers[i].self_attn"""
        return model.model.layers[layer_idx].self_attn

    def _register_hooks(self, model: nn.Module):
        """Register hooks on self_attn to capture K/V pre-RoPE.

        Hooks fire on the self_attn forward pre-hook where the input is
        the NORMED hidden state (post-input_layernorm). This is the same
        input that q_proj/k_proj/v_proj use inside self_attn, so the
        captured K/V are from the correct distribution.

        Using layer.register_forward_hook (old approach) gave input[0] =
        un-normed residual stream, causing Q (normed) @ K (un-normed)
        dot products to be meaningless garbage.
        """
        for layer_idx in self.layer_indices:
            attn = self._get_attention_module(model, layer_idx)

            def make_attn_hook(l_idx, attn_module, capture_hs):
                def hook(module, args, kwargs):
                    # self_attn pre_hook: args[0] or kwargs['hidden_states']
                    # is the normed hidden state (post-input_layernorm, pre-RoPE)
                    if 'hidden_states' in kwargs:
                        hidden = kwargs['hidden_states']
                    elif isinstance(args, tuple) and len(args) > 0:
                        hidden = args[0]
                    else:
                        return
                    with torch.no_grad():
                        k = attn_module.k_proj(hidden).detach()
                        v = attn_module.v_proj(hidden).detach()
                    self._buffer[l_idx] = [k, v]
                    if capture_hs:
                        self._hs_buffer[l_idx] = hidden.detach()
                return hook

            h = attn.register_forward_pre_hook(
                make_attn_hook(layer_idx, attn, self.capture_hidden_states),
                with_kwargs=True,
            )
            self._hooks.append(h)

    def get_buffered_hidden_states(self) -> dict:
        """Return captured hidden states. {layer_idx: Tensor[1, T, d_model]}."""
        return {l: hs for l, hs in self._hs_buffer.items() if hs is not None}

    def get_buffered_kv(self) -> dict:
        """Return captured K/V pairs. {layer_idx: (K, V)}."""
        result = {}
        for l_idx, kv in self._buffer.items():
            k, v = kv
            if k is not None and v is not None:
                result[l_idx] = (k, v)
        return result

    def clear_buffer(self):
        """Clear the capture buffer."""
        self._buffer.clear()
        self._hs_buffer.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._buffer.clear()
