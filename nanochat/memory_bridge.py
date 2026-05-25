# nanochat/memory_bridge.py
"""MemoryBridge: RecursiveMAS-style latent memory prefix on a frozen backbone.

Idea (arXiv 2604.25917, "Recursive Multi-Agent Systems"): a lightweight,
trainable bridge transfers latent state into a *frozen* model at the input
embedding level, with a projection that keeps the latent in-distribution.

Here we reuse that mechanism for memory:

  1. Encode a memory string by running the frozen backbone over
     ``[memory_tokens, K learnable gist tokens]`` and taking the last-layer
     hidden states at the K gist positions  ->  H_mem in [K, d_model].
  2. Project them with the RecursiveMAS *outer* form
        R(h) = W3 h + W2 . GELU(W1 h)
     into a soft prefix  L in [K, d_model].
  3. Prepend L to the query's input embeddings and generate.

Only this module is trained; the backbone weights stay frozen. The module is
backbone-agnostic: it consumes/produces ``d_model`` vectors, so it works for
standard attention stacks and for hybrid (e.g. Gated DeltaNet) ones alike.

The encode/inject/generate orchestration lives in the training/eval script,
which owns the model handle. This module is pure and unit-testable.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MemoryBridge(nn.Module):
    """Trainable bridge: K gist token embeddings + an outer projection R(.).

    Args:
        d_model: backbone hidden / embedding dimension.
        k_gist: number of latent (gist) tokens per memory.
        hidden_mult: width multiplier for the projection's bottleneck.
        init_std: std for the (non-zero) normal init of weights and gist tokens.
    """

    def __init__(self, d_model: int, k_gist: int = 16,
                 hidden_mult: int = 2, init_std: float = 0.02):
        super().__init__()
        self.d_model = d_model
        self.k_gist = k_gist
        hidden = d_model * hidden_mult

        # Learnable gist tokens appended (as embeddings) to the memory before
        # the frozen forward pass; their last-layer hidden states are the
        # compressed memory read-out.
        self.gist_tokens = nn.Parameter(torch.randn(k_gist, d_model) * init_std)

        # RecursiveMAS outer projection: R(h) = W3 h + W2 . GELU(W1 h)
        self.w1 = nn.Linear(d_model, hidden, bias=True)
        self.w2 = nn.Linear(hidden, d_model, bias=True)
        self.w3 = nn.Linear(d_model, d_model, bias=True)
        self.act = nn.GELU()

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        # Never zero-init: a zero weight kills the gradient to upstream factors
        # via the chain rule (recurring lesson in this codebase). Biases at 0
        # are fine — they sit in an additive position and still receive grad.
        for lin in (self.w1, self.w2, self.w3):
            nn.init.normal_(lin.weight, std=std)
            nn.init.zeros_(lin.bias)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Map hidden states into the latent prefix space.

        Args:
            h: [..., d_model] hidden states (typically [B, K, d] or [K, d]).
        Returns:
            [..., d_model] projected latent prefix.
        """
        return self.w3(h) + self.w2(self.act(self.w1(h)))

    def gist_embeds(self, batch_size: int = 1) -> torch.Tensor:
        """Gist token embeddings to append to memory embeddings before forward.

        Returns:
            [batch_size, k_gist, d_model] (an expanded view of the parameter).
        """
        return self.gist_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, h_gist: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`project` so the module is callable."""
        return self.project(h_gist)


def answer_logit_span(prefix_len: int, query_len: int, answer_len: int
                      ) -> tuple[int, int]:
    """Index span of the logits that should predict the answer tokens.

    The decoder consumes ``[prefix(prefix_len); query(query_len); answer(answer_len)]``
    as ``inputs_embeds``. In a causal LM the logit at position ``i`` predicts the
    token at position ``i + 1``. The first answer token sits at position
    ``prefix_len + query_len``, so the logit predicting it is at
    ``prefix_len + query_len - 1``. The span covering all answer tokens is
    therefore ``[start, start + answer_len)`` with ``start = prefix_len + query_len - 1``.

    Returns:
        (start, end) half-open logit indices; slice ``logits[..., start:end, :]``
        aligns one-to-one with the ``answer_len`` answer token ids.
    """
    if min(prefix_len, query_len, answer_len) < 0:
        raise ValueError("lengths must be non-negative")
    if query_len == 0 and prefix_len == 0:
        raise ValueError("need at least one conditioning token before the answer")
    start = prefix_len + query_len - 1
    return start, start + answer_len
