# tests/test_cellmem_v2.py
"""
Unit tests for CellMem v2. Run as:
python -m pytest tests/test_cellmem_v2.py -v
"""
import torch
import pytest
from nanochat.cellmem_v2 import CellMemConfig, MemoryStore


class TestMemoryStoreInit:
    def test_default_config(self):
        cfg = CellMemConfig()
        assert cfg.enabled is False
        assert cfg.n_slots == 64
        assert cfg.write_strategy == "per_token"

    def test_init_empty(self):
        cfg = CellMemConfig(enabled=True)
        store = MemoryStore(cfg, d_model=768)
        assert store.active_count == 0
        assert store.write_ptr == 0
        assert store.vectors.shape == (64, 768)

    def test_read_empty_returns_none(self):
        cfg = CellMemConfig(enabled=True)
        store = MemoryStore(cfg, d_model=768)
        result = store.read()
        assert result is None


class TestMemoryStoreWrite:
    def test_write_single(self):
        cfg = CellMemConfig(enabled=True, n_slots=4)
        store = MemoryStore(cfg, d_model=16)
        vec = torch.randn(16)
        store.write(vec, surprise=5.0)
        assert store.active_count == 1
        assert store.write_ptr == 1

    def test_write_then_read(self):
        cfg = CellMemConfig(enabled=True, n_slots=4)
        store = MemoryStore(cfg, d_model=16)
        vec = torch.randn(16)
        store.write(vec, surprise=5.0)
        result = store.read()
        assert result is not None
        vectors, mask = result
        assert vectors.shape == (4, 16)
        assert mask.sum().item() == 1  # only 1 active slot

    def test_write_fills_slots_sequentially(self):
        cfg = CellMemConfig(enabled=True, n_slots=4)
        store = MemoryStore(cfg, d_model=8)
        for i in range(3):
            store.write(torch.randn(8), surprise=float(i))
        assert store.active_count == 3
        assert store.write_ptr == 3

    def test_write_clamps_vector_norm(self):
        cfg = CellMemConfig(enabled=True, n_slots=4)
        store = MemoryStore(cfg, d_model=8)
        big_vec = torch.ones(8) * 1000  # huge norm
        store.write(big_vec, surprise=5.0)
        stored = store.vectors[0]
        assert stored.norm().item() == pytest.approx(50.0, abs=0.1)  # clamped to max_vector_norm
