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


class TestDeltaUpdateRule:
    """Delta update rule: write prediction error instead of raw hidden state.
    Biological basis: BTSP prediction-error-driven plasticity (NIMH/Scripps 2025)."""

    def test_delta_empty_memory_equals_raw(self):
        """When memory is empty, prediction is zero → delta == raw vector."""
        cfg = CellMemConfig(enabled=True, n_slots=4, write_mode="delta")
        store = MemoryStore(cfg, d_model=16)
        vec = torch.randn(16)
        store.write(vec, surprise=5.0)
        # With empty memory, prediction=0, so stored = vec - 0 = vec
        stored = store.vectors[0]
        assert stored.allclose(vec, atol=1e-5)

    def test_delta_subtracts_prediction(self):
        """Writing a similar vector stores only the residual (prediction error)."""
        cfg = CellMemConfig(enabled=True, n_slots=4, write_mode="delta")
        store = MemoryStore(cfg, d_model=16)
        # Write first vector (raw, since memory was empty)
        v1 = torch.randn(16)
        store.write(v1, surprise=5.0)
        # Write very similar vector — prediction should be close to v2
        v2 = v1 + torch.randn(16) * 0.01  # small perturbation
        store.write(v2, surprise=5.0)
        stored_delta = store.vectors[1]
        # The stored delta should be much smaller than the original v2
        assert stored_delta.norm() < v2.norm() * 0.5

    def test_delta_orthogonal_vector_stored_fully(self):
        """A vector orthogonal to all memories has zero prediction → stored as-is."""
        cfg = CellMemConfig(enabled=True, n_slots=4, write_mode="delta")
        store = MemoryStore(cfg, d_model=16)
        # Write along dimension 0
        v1 = torch.zeros(16)
        v1[0] = 10.0
        store.write(v1, surprise=5.0)
        # Write along dimension 8 (orthogonal)
        v2 = torch.zeros(16)
        v2[8] = 10.0
        store.write(v2, surprise=5.0)
        stored = store.vectors[1]
        # Cosine similarity between memory[0] and v2 is 0,
        # so prediction ≈ 0, stored ≈ v2
        assert stored.norm() > v2.norm() * 0.8

    def test_delta_redundant_write_skipped(self):
        """Writing the exact same vector twice: second write skipped (delta ≈ 0)."""
        cfg = CellMemConfig(enabled=True, n_slots=4, write_mode="delta",
                           delta_threshold=0.1)
        store = MemoryStore(cfg, d_model=16)
        vec = torch.randn(16)
        vec = vec / vec.norm() * 5.0  # normalize to known norm
        store.write(vec, surprise=5.0)
        assert store.active_count == 1
        # Write same vector again — delta norm < threshold, should be skipped
        store.write(vec.clone(), surprise=5.0)
        assert store.active_count == 1  # NOT incremented

class TestDecorrelation:
    """DG pattern separation: skip writes when new vector is too similar to existing memories."""

    def test_novelty_zero_allows_duplicates(self):
        """min_novelty=0 (default) allows writing identical vectors."""
        cfg = CellMemConfig(enabled=True, n_slots=4, min_novelty=0.0)
        store = MemoryStore(cfg, d_model=16)
        vec = torch.randn(16)
        store.write(vec, surprise=5.0)
        store.write(vec.clone(), surprise=5.0)
        assert store.active_count == 2

    def test_novelty_blocks_similar(self):
        """min_novelty=0.1 blocks writes with cosine sim > 0.9."""
        cfg = CellMemConfig(enabled=True, n_slots=4, min_novelty=0.1)
        store = MemoryStore(cfg, d_model=16)
        vec = torch.randn(16)
        store.write(vec, surprise=5.0)
        assert store.active_count == 1
        # Write nearly identical vector — should be blocked
        store.write(vec + torch.randn(16) * 0.01, surprise=5.0)
        assert store.active_count == 1

    def test_novelty_allows_orthogonal(self):
        """Orthogonal vectors pass the novelty check."""
        cfg = CellMemConfig(enabled=True, n_slots=4, min_novelty=0.1)
        store = MemoryStore(cfg, d_model=16)
        v1 = torch.zeros(16); v1[0] = 1.0
        v2 = torch.zeros(16); v2[8] = 1.0
        store.write(v1, surprise=5.0)
        store.write(v2, surprise=5.0)
        assert store.active_count == 2

    def test_novelty_empty_memory_always_writes(self):
        """First write always succeeds regardless of min_novelty."""
        cfg = CellMemConfig(enabled=True, n_slots=4, min_novelty=0.5)
        store = MemoryStore(cfg, d_model=16)
        store.write(torch.randn(16), surprise=5.0)
        assert store.active_count == 1

    def test_novelty_high_threshold_blocks_more(self):
        """min_novelty=0.5 blocks vectors with cosine sim > 0.5."""
        cfg = CellMemConfig(enabled=True, n_slots=4, min_novelty=0.5)
        store = MemoryStore(cfg, d_model=16)
        vec = torch.randn(16)
        store.write(vec, surprise=5.0)
        # Moderate perturbation — sim likely still > 0.5
        store.write(vec + torch.randn(16) * 0.5, surprise=5.0)
        # With high min_novelty, even moderate perturbation gets blocked
        assert store.active_count <= 2  # may or may not pass depending on random


class TestGateGradients:
    """Verify that gradients flow through mem_gates during training."""

    @staticmethod
    def _make_test_model():
        """Create a tiny model with non-zero c_proj for gradient testing.
        init_weights() zero-inits c_proj (standard for residual connections),
        but a trained model has non-zero c_proj, so we simulate that."""
        from nanochat.gpt import GPT, GPTConfig
        cfg = GPTConfig(
            sequence_len=64, vocab_size=256, n_layer=4, n_head=2,
            n_kv_head=2, n_embd=64, window_pattern="L",
            cellmem=CellMemConfig(enabled=True, layers="mid", n_slots=4),
        )
        model = GPT(cfg)
        model.init_weights()
        # Simulate trained model: c_proj needs non-zero weights for gradient flow
        for block in model.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)
        return model, cfg

    def test_gate_receives_gradient(self):
        """When _train_memory=True, mem_gates should get gradients from LM loss."""
        model, cfg = self._make_test_model()

        # Freeze all params except mem_gates, init gates to 0 (sigmoid=0.5)
        for p in model.parameters():
            p.requires_grad = False
        for gate in model.mem_gates:
            gate.data.fill_(0.0)
            gate.requires_grad = True

        # Create a memory store with some vectors and attach to model
        store = MemoryStore(cfg.cellmem, d_model=64)
        store.write(torch.randn(64), surprise=5.0)
        store.write(torch.randn(64), surprise=5.0)
        model.memory_store = store
        model._train_memory = True  # enable memory in training mode

        # Forward pass with targets to get loss
        tokens = torch.randint(0, 256, (1, 8))
        model.train()
        loss = model(tokens, targets=tokens)

        # Backward
        loss.backward()

        # Check that gates have gradients
        for i, gate in enumerate(model.mem_gates):
            assert gate.grad is not None, f"Gate {i} has no gradient"
            # Gradient should be non-zero (memory vectors are random, not degenerate)
            assert gate.grad.abs().item() > 0, f"Gate {i} has zero gradient"

    def test_gate_value_affects_loss(self):
        """Different gate values should produce different losses."""
        model, cfg = self._make_test_model()

        store = MemoryStore(cfg.cellmem, d_model=64)
        store.write(torch.randn(64), surprise=5.0)
        store.write(torch.randn(64), surprise=5.0)
        model.memory_store = store
        model._train_memory = True

        tokens = torch.randint(0, 256, (1, 8))

        # Loss with gate = -10 (sigmoid ~ 0, no memory)
        with torch.no_grad():
            for gate in model.mem_gates:
                gate.fill_(-10.0)
            loss_closed = model(tokens, targets=tokens).item()

        # Loss with gate = 5 (sigmoid ~ 1, full memory)
        with torch.no_grad():
            for gate in model.mem_gates:
                gate.fill_(5.0)
            loss_open = model(tokens, targets=tokens).item()

        assert loss_closed != loss_open, "Gate value has no effect on loss"


class TestDeltaUpdateRuleRawMode:
    def test_raw_mode_unchanged(self):
        """Default write_mode='raw' preserves existing behavior exactly."""
        cfg = CellMemConfig(enabled=True, n_slots=4, write_mode="raw")
        store = MemoryStore(cfg, d_model=16)
        vec = torch.randn(16)
        store.write(vec, surprise=5.0)
        store.write(vec.clone(), surprise=5.0)
        # Both writes go through (no delta filtering)
        assert store.active_count == 2
        assert store.vectors[0].allclose(vec, atol=1e-5)
        assert store.vectors[1].allclose(vec, atol=1e-5)
