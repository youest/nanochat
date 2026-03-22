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


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        cfg = CellMemConfig(enabled=True, n_slots=4, decay_factor=1.0)
        store = MemoryStore(cfg, d_model=8)
        vec = torch.randn(8)
        store.write(vec, surprise=5.0)
        save_path = tmp_path / "memory.pt"
        store.save(save_path)
        assert save_path.exists()
        store2 = MemoryStore(cfg, d_model=8)
        store2.load(save_path)
        assert store2.active_count == 1
        assert store2.vectors[0].allclose(vec, atol=1e-4)

    def test_load_applies_decay(self, tmp_path):
        cfg = CellMemConfig(enabled=True, n_slots=4, decay_factor=0.5)
        store = MemoryStore(cfg, d_model=8)
        vec = torch.ones(8)
        store.write(vec, surprise=5.0)
        save_path = tmp_path / "memory.pt"
        store.save(save_path)
        store2 = MemoryStore(cfg, d_model=8)
        store2.load(save_path)
        expected = vec * 0.5
        assert store2.vectors[0].allclose(expected, atol=1e-6)

    def test_snapshot_creates_timestamped_file(self, tmp_path):
        cfg = CellMemConfig(enabled=True, n_slots=4, memory_dir=str(tmp_path))
        store = MemoryStore(cfg, d_model=8)
        store.write(torch.randn(8), surprise=5.0)
        store.snapshot()
        snaps = list(tmp_path.glob("snapshot_*.pt"))
        assert len(snaps) == 1

    def test_snapshot_max_retention(self, tmp_path):
        import time
        cfg = CellMemConfig(enabled=True, n_slots=4, max_snapshots=2, memory_dir=str(tmp_path))
        store = MemoryStore(cfg, d_model=8)
        store.write(torch.randn(8), surprise=5.0)
        for _ in range(4):
            store.snapshot()
            time.sleep(0.002)  # ensure unique timestamps
        snaps = sorted(tmp_path.glob("snapshot_*.pt"))
        assert len(snaps) == 2  # only 2 retained

    def test_save_format_version(self, tmp_path):
        cfg = CellMemConfig(enabled=True, n_slots=4)
        store = MemoryStore(cfg, d_model=8)
        store.write(torch.randn(8), surprise=5.0)
        save_path = tmp_path / "memory.pt"
        store.save(save_path)
        data = torch.load(save_path, weights_only=False)
        assert data["version"] == 2
        assert "saved_at" in data
        assert "config" in data


class TestSurpriseCalculator:
    def test_per_token_computes_loss(self):
        from nanochat.cellmem_v2 import SurpriseCalculator
        cfg = CellMemConfig(enabled=True, surprise_threshold=2.0, write_strategy="per_token")
        calc = SurpriseCalculator(cfg)
        logits = torch.zeros(1, 4, 8)
        logits[0, :, 0] = 10.0  # strongly predict token 0
        targets = torch.tensor([[0, 0, 5, 0]])  # token at pos 2 is surprising
        surprises = calc.compute_surprise(logits, targets)
        assert surprises.shape == (1, 4)
        assert surprises[0, 2] > surprises[0, 0]

    def test_per_token_threshold_filter(self):
        from nanochat.cellmem_v2 import SurpriseCalculator
        cfg = CellMemConfig(enabled=True, surprise_threshold=2.0, write_strategy="per_token")
        calc = SurpriseCalculator(cfg)
        logits = torch.zeros(1, 4, 8)
        logits[0, :, 0] = 10.0
        targets = torch.tensor([[0, 0, 5, 0]])
        surprises = calc.compute_surprise(logits, targets)
        write_mask = calc.get_write_mask(surprises)
        assert write_mask.shape == (1, 4)
        assert write_mask[0, 2].item() is True
        assert write_mask[0, 0].item() is False

    def test_chunk_strategy_averages(self):
        from nanochat.cellmem_v2 import SurpriseCalculator
        cfg = CellMemConfig(enabled=True, surprise_threshold=2.0,
                           write_strategy="chunk", chunk_size=2)
        calc = SurpriseCalculator(cfg)
        logits = torch.zeros(1, 4, 8)
        logits[0, :, 0] = 10.0
        targets = torch.tensor([[5, 5, 0, 0]])  # first chunk surprising, second not
        surprises = calc.compute_surprise(logits, targets)
        chunks = calc.get_chunk_surprises(surprises)
        assert len(chunks) == 2
        assert chunks[0]["mean_surprise"] > chunks[1]["mean_surprise"]

    def test_deduplication_tracking(self):
        from nanochat.cellmem_v2 import SurpriseCalculator
        cfg = CellMemConfig(enabled=True, surprise_threshold=0.0, write_strategy="per_token")
        calc = SurpriseCalculator(cfg)
        logits = torch.zeros(1, 4, 8)
        targets = torch.zeros(1, 4, dtype=torch.long)
        surprises = calc.compute_surprise(logits, targets)
        mask1 = calc.get_write_mask(surprises, last_written_pos=1)
        assert mask1[0, 0].item() is False
        assert mask1[0, 1].item() is False
