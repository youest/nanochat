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


class TestMemoryStoreEviction:
    def test_eviction_when_full(self):
        """When full, the slot with lowest surprise is overwritten."""
        cfg = CellMemConfig(enabled=True, n_slots=3)
        store = MemoryStore(cfg, d_model=8)
        store.write(torch.ones(8) * 1, surprise=2.0)  # slot 0, low surprise
        store.write(torch.ones(8) * 2, surprise=8.0)  # slot 1, high surprise
        store.write(torch.ones(8) * 3, surprise=5.0)  # slot 2, medium surprise
        assert store.active_count == 3
        # Write a 4th: should evict slot 0 (lowest surprise=2.0)
        store.write(torch.ones(8) * 4, surprise=6.0)
        assert store.active_count == 3  # still 3
        assert store.surprise[0].item() == 6.0  # slot 0 was overwritten
        assert store.vectors[0].allclose(torch.ones(8) * 4)

    def test_all_slots_active_after_fill(self):
        cfg = CellMemConfig(enabled=True, n_slots=2)
        store = MemoryStore(cfg, d_model=4)
        store.write(torch.randn(4), surprise=1.0)
        store.write(torch.randn(4), surprise=2.0)
        _, mask = store.read()
        assert mask.all()  # all slots active when full

    def test_age_increments(self):
        cfg = CellMemConfig(enabled=True, n_slots=4)
        store = MemoryStore(cfg, d_model=4)
        store.write(torch.randn(4), surprise=5.0)
        store.age_all(tokens_processed=100)
        assert store.age[0].item() == 100
