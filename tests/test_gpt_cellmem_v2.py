# tests/test_gpt_cellmem_v2.py
"""
Integration tests for GPT + CellMem v2. Run as:
python -m pytest tests/test_gpt_cellmem_v2.py -v
"""
import torch
import pytest


class TestAttentionWithMemory:
    def test_forward_without_memory_unchanged(self):
        """When mem_kv=None, output identical to original."""
        from nanochat.gpt import CausalSelfAttention, GPTConfig, norm
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
        attn = CausalSelfAttention(cfg, layer_idx=0)
        torch.manual_seed(42)
        for p in attn.parameters():
            p.data.normal_(0, 0.02)
        x = torch.randn(1, 8, 32)
        cos = torch.ones(1, 8, 1, 8)
        sin = torch.zeros(1, 8, 1, 8)
        y1 = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None)
        y2 = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None,
                  mem_kv=None, mem_gate=None)
        assert torch.allclose(y1, y2, atol=1e-6)

    def test_forward_with_memory_gate_zero(self):
        """With mem_gate near 0, output nearly identical to no-memory."""
        from nanochat.gpt import CausalSelfAttention, GPTConfig, norm
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
        attn = CausalSelfAttention(cfg, layer_idx=0)
        torch.manual_seed(42)
        for p in attn.parameters():
            p.data.normal_(0, 0.02)
        x = torch.randn(1, 8, 32)
        cos = torch.ones(1, 8, 1, 8)
        sin = torch.zeros(1, 8, 1, 8)
        K_mem = torch.randn(1, 4, 2, 16)  # [B, K, Hkv, D]
        V_mem = torch.randn(1, 4, 2, 16)
        gate = torch.sigmoid(torch.tensor([-10.0]))  # near 0
        y_no = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None)
        y_mem = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None,
                     mem_kv=(K_mem, V_mem), mem_gate=gate)
        assert torch.allclose(y_no, y_mem, atol=1e-3)

    def test_forward_with_memory_gate_one(self):
        """With mem_gate=1, output should differ from no-memory."""
        from nanochat.gpt import CausalSelfAttention, GPTConfig, norm
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
        attn = CausalSelfAttention(cfg, layer_idx=0)
        torch.manual_seed(42)
        for p in attn.parameters():
            p.data.normal_(0, 0.02)
        x = torch.randn(1, 8, 32)
        cos = torch.ones(1, 8, 1, 8)
        sin = torch.zeros(1, 8, 1, 8)
        K_mem = torch.randn(1, 4, 2, 16)
        V_mem = torch.randn(1, 4, 2, 16)
        gate = torch.tensor([1.0])
        y_no = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None)
        y_mem = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None,
                     mem_kv=(K_mem, V_mem), mem_gate=gate)
        assert not torch.allclose(y_no, y_mem, atol=0.01)

    def test_block_forward_passes_through_memory(self):
        """Block.forward passes mem_kv and mem_gate to attention."""
        from nanochat.gpt import Block, GPTConfig, norm
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
        block = Block(cfg, layer_idx=0)
        torch.manual_seed(42)
        for p in block.parameters():
            p.data.normal_(0, 0.02)
        x = torch.randn(1, 8, 32)
        cos = torch.ones(1, 8, 1, 8)
        sin = torch.zeros(1, 8, 1, 8)
        # Without memory
        y_no = block(x, ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None)
        # With memory and gate=1 should differ
        K_mem = torch.randn(1, 4, 2, 16)
        V_mem = torch.randn(1, 4, 2, 16)
        gate = torch.tensor([1.0])
        y_mem = block(x, ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None,
                      mem_kv=(K_mem, V_mem), mem_gate=gate)
        assert not torch.allclose(y_no, y_mem, atol=0.01)
