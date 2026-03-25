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


class TestMixedTrainingData:
    def test_negative_data_has_irrelevant_memory(self):
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=50, seed=42)
        negatives = [d for d in data if d["type"] == "negative"]
        assert len(negatives) > 0
        for neg in negatives:
            assert "context" in neg
            assert "query" in neg
            assert "answer" in neg
            assert neg["type"] == "negative"

    def test_poisoned_data_has_wrong_memory(self):
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=50, seed=42)
        poisoned = [d for d in data if d["type"] == "poisoned"]
        assert len(poisoned) > 0
        for p in poisoned:
            assert "context" in p
            assert "query" in p
            assert "answer" in p
            assert "wrong_context" in p

    def test_data_ratios(self):
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=100, seed=42)
        counts = {"positive": 0, "negative": 0, "poisoned": 0}
        for d in data:
            counts[d["type"]] += 1
        assert counts["positive"] >= 30
        assert counts["negative"] >= 30
        assert counts["poisoned"] >= 10

    def test_positive_data_matches_original_format(self):
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=20, seed=42)
        positives = [d for d in data if d["type"] == "positive"]
        assert len(positives) > 0
        for p in positives:
            assert "context" in p
            assert "query" in p
            assert "answer" in p
