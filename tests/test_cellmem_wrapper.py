"""Tests for CellMemWrapper components (no GPU required)."""
import torch
import torch.nn.functional as F
import pytest


class TestCellMemWrapperUsesContentGate:
    def test_content_gate_produces_per_token_values(self):
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=64)
        h_local = torch.randn(2, 8, 64)
        h_mem = torch.randn(2, 8, 64)
        g = gate(h_local, h_mem)
        assert g.shape == (2, 8, 1)

    def test_content_gate_trainable_param_count(self):
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=256)
        n_params = sum(p.numel() for p in gate.parameters())
        assert n_params > 1000
        assert n_params < 200_000

    def test_rms_norm_equalizes_attention(self):
        from nanochat.cellmem_v2 import MemoryRMSNorm
        d = 64
        norm = MemoryRMSNorm(d_model=d)
        mem = torch.randn(5, d)
        mem[0] *= 100
        query = torch.randn(1, d)
        scores_raw = (query @ mem.T) / (d ** 0.5)
        attn_raw = torch.softmax(scores_raw, dim=-1)
        # clamp to avoid log(0)=-inf producing nan entropy
        entropy_raw = -(attn_raw * attn_raw.clamp(min=1e-9).log()).sum()
        mem_normed = norm(mem)
        scores_normed = (query @ mem_normed.T) / (d ** 0.5)
        attn_normed = torch.softmax(scores_normed, dim=-1)
        entropy_normed = -(attn_normed * attn_normed.clamp(min=1e-9).log()).sum()
        assert entropy_normed > entropy_raw
