"""TDD tests for MemoryBridge (RecursiveMAS-style latent memory prefix).

The bridge projects last-layer hidden states (at K learnable gist positions)
into a soft prefix that is prepended to the query's input embeddings.

These tests exercise the pure module (no model download): shapes, gradient
flow, no zero-init, non-identity projection.
"""
import torch
import pytest

from nanochat.memory_bridge import MemoryBridge, answer_logit_span


D = 32
K = 4


@pytest.fixture
def bridge():
    return MemoryBridge(d_model=D, k_gist=K)


class TestShapes:
    def test_project_preserves_d_model(self, bridge):
        h = torch.randn(2, K, D)
        out = bridge.project(h)
        assert out.shape == (2, K, D)

    def test_project_unbatched(self, bridge):
        h = torch.randn(K, D)
        out = bridge.project(h)
        assert out.shape == (K, D)

    def test_gist_tokens_shape(self, bridge):
        assert bridge.gist_tokens.shape == (K, D)

    def test_gist_embeds_batch_expand(self, bridge):
        emb = bridge.gist_embeds(batch_size=3)
        assert emb.shape == (3, K, D)
        # all rows identical (expanded view of the same parameter)
        assert torch.allclose(emb[0], emb[1]) and torch.allclose(emb[1], emb[2])


class TestInitialization:
    def test_no_zero_init_weights(self, bridge):
        # Zero-init blocks gradients via chain rule (project README lesson).
        for name in ("w1", "w2", "w3"):
            w = getattr(bridge, name).weight
            assert w.norm().item() > 0, f"{name}.weight is zero-initialized"

    def test_gist_tokens_nonzero(self, bridge):
        assert bridge.gist_tokens.norm().item() > 0

    def test_gist_tokens_trainable(self, bridge):
        assert bridge.gist_tokens.requires_grad
        assert any(p is bridge.gist_tokens for p in bridge.parameters())


class TestProjection:
    def test_not_identity(self, bridge):
        h = torch.randn(2, K, D)
        out = bridge.project(h)
        assert not torch.allclose(out, h), "projection collapsed to identity"

    def test_output_finite(self, bridge):
        h = torch.randn(2, K, D) * 10
        out = bridge.project(h)
        assert torch.isfinite(out).all()


class TestGradientFlow:
    def test_grad_reaches_all_projection_weights(self, bridge):
        h = torch.randn(2, K, D, requires_grad=False)
        loss = bridge.project(h).pow(2).sum()
        loss.backward()
        for name in ("w1", "w2", "w3"):
            g = getattr(bridge, name).weight.grad
            assert g is not None, f"no grad for {name}"
            assert g.norm().item() > 0, f"zero grad for {name}"

    def test_grad_reaches_gist_tokens(self, bridge):
        # Gist tokens reach the loss through a downstream consumer of their embeds.
        emb = bridge.gist_embeds(batch_size=1)          # [1, K, D]
        loss = bridge.project(emb).pow(2).sum()
        loss.backward()
        assert bridge.gist_tokens.grad is not None
        assert bridge.gist_tokens.grad.norm().item() > 0


class TestParamBudget:
    def test_param_count_is_small(self, bridge):
        # Bridge is the only trainable piece; backbone stays frozen elsewhere.
        n = sum(p.numel() for p in bridge.parameters())
        # gist (K*D) + w1 (D*h+h) + w2 (h*D+D) + w3 (D*D+D), all tiny for D=32
        assert 0 < n < 50_000


class TestAnswerLogitSpan:
    def test_basic_alignment(self):
        # prefix=16, query=10, answer=3 -> first answer token at pos 26,
        # predicted by logit at pos 25; span covers 3 logits.
        start, end = answer_logit_span(16, 10, 3)
        assert (start, end) == (25, 28)
        assert end - start == 3

    def test_span_width_equals_answer_len(self):
        for p, q, a in [(16, 5, 1), (4, 20, 7), (1, 1, 1)]:
            start, end = answer_logit_span(p, q, a)
            assert end - start == a

    def test_no_conditioning_raises(self):
        with pytest.raises(ValueError):
            answer_logit_span(0, 0, 3)

    def test_prefix_only_conditioning_ok(self):
        # prefix alone can condition the answer (query_len=0)
        start, end = answer_logit_span(8, 0, 2)
        assert (start, end) == (7, 9)
