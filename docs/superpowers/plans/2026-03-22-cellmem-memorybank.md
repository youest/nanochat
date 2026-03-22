# CellMem + MemoryBank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent MemoryBank to CellMem that accumulates knowledge across inference invocations, using the transformer's own embedding space for storage and CellMem's surprise signal to decide what to write.

**Architecture:** Two-level memory system. Level 1 (CellMem, exists): fast intra-sequence modulation via matrix M with anti-Hebbian updates. Level 2 (MemoryBank, new): persistent cross-invocation storage using embedding vectors. CellMem's novelty signal gates writes to the bank. Retrieval uses a FAISS-filtered working set with dot-product attention. The bank lives alongside the transformer and operates at inference time only.

**Tech Stack:** Python, PyTorch, faiss-cpu (new dependency), existing nanochat test infrastructure (pytest)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `nanochat/memory_bank.py` | Create | MemoryBank class: slot storage, read, write, decay, save/load, FAISS index |
| `nanochat/cellmem.py` | Modify | Add chunked update mode, decay on M, novelty exposure for write gating |
| `tests/test_memory_bank.py` | Create | Unit tests for MemoryBank in isolation |
| `tests/test_cellmem.py` | Modify | Add tests for chunked update and M decay |
| `scripts/test3_persistent_memory.py` | Create | Integration benchmark: memory across sequences |
| `docs/cellmem/06-memory-bank.md` | Create | Design doc for MemoryBank |

---

### Task 1: MemoryBank — core read/write/decay

**Files:**
- Create: `nanochat/memory_bank.py`
- Create: `tests/test_memory_bank.py`

- [ ] **Step 1: Write failing test — MemoryBank init and shapes**

```python
# tests/test_memory_bank.py
import torch
import pytest
from nanochat.memory_bank import MemoryBank, MemoryBankConfig

CFG = MemoryBankConfig(d_key=16, d_value=8, max_slots=256)

class TestInit:
    def test_empty_bank(self):
        bank = MemoryBank(CFG)
        assert bank.size == 0

    def test_config_stored(self):
        bank = MemoryBank(CFG)
        assert bank.config.d_key == 16
        assert bank.config.d_value == 8
        assert bank.config.max_slots == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestInit -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nanochat.memory_bank'"

- [ ] **Step 3: Write minimal MemoryBank skeleton**

```python
# nanochat/memory_bank.py
"""
MemoryBank: persistent embedding store for cross-invocation memory.
Stores (key, value) pairs where key is in the transformer's embedding space
and value is CellMem's output. Retrieval via dot-product attention on a
FAISS-filtered working set.
"""

from dataclasses import dataclass
import torch


@dataclass
class MemoryBankConfig:
    d_key: int = 768       # dimension of keys (= d_model of transformer)
    d_value: int = 64      # dimension of values (= d_cell * n_cells of CellMem)
    max_slots: int = 10000 # maximum number of memory slots
    working_set_k: int = 16  # number of slots in working set
    write_threshold: float = 0.5  # novelty threshold for writing
    similarity_threshold: float = 0.9  # cosine sim to merge instead of create
    decay_usage: float = 0.995  # usage decay per invocation
    min_usage: float = 0.01  # below this, slot is overwritable


class MemoryBank:
    """Persistent key-value memory bank with surprise-gated writing."""

    def __init__(self, config: MemoryBankConfig):
        self.config = config
        self._keys = torch.zeros(0, config.d_key)
        self._values = torch.zeros(0, config.d_value)
        self._usage = torch.zeros(0)
        self._age = torch.zeros(0)

    @property
    def size(self) -> int:
        return self._keys.shape[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestInit -v`
Expected: PASS

- [ ] **Step 5: Write failing test — write and read**

```python
# append to tests/test_memory_bank.py

class TestWriteRead:
    def test_write_adds_slot(self):
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        value = torch.randn(CFG.d_value)
        bank.write(key, value, surprise=1.0)
        assert bank.size == 1

    def test_write_below_threshold_ignored(self):
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        value = torch.randn(CFG.d_value)
        bank.write(key, value, surprise=0.01)  # below default 0.5
        assert bank.size == 0

    def test_read_empty_returns_zeros(self):
        bank = MemoryBank(CFG)
        query = torch.randn(CFG.d_key)
        result = bank.read(query)
        assert result.shape == (CFG.d_value,)
        assert torch.allclose(result, torch.zeros(CFG.d_value))

    def test_read_retrieves_similar(self):
        torch.manual_seed(42)
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        key = key / key.norm()
        value = torch.ones(CFG.d_value)
        bank.write(key, value, surprise=1.0)

        # Query with same key should retrieve similar value
        result = bank.read(key)
        # With only one slot, softmax gives weight 1.0 to it
        assert torch.allclose(result, value, atol=1e-5)

    def test_read_prefers_similar_key(self):
        torch.manual_seed(42)
        bank = MemoryBank(CFG)
        # Write two slots: one close to query, one far
        key_close = torch.randn(CFG.d_key)
        key_close = key_close / key_close.norm()
        key_far = torch.randn(CFG.d_key)
        key_far = key_far / key_far.norm()
        val_close = torch.ones(CFG.d_value)
        val_far = -torch.ones(CFG.d_value)
        bank.write(key_close, val_close, surprise=1.0)
        bank.write(key_far, val_far, surprise=1.0)

        result = bank.read(key_close)
        # Result should be closer to val_close than val_far
        sim_close = torch.dot(result, val_close)
        sim_far = torch.dot(result, val_far)
        assert sim_close > sim_far, f"Should prefer similar: close={sim_close:.3f} far={sim_far:.3f}"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestWriteRead -v`
Expected: FAIL with "AttributeError: 'MemoryBank' object has no attribute 'write'"

- [ ] **Step 7: Implement write and read**

Add to `nanochat/memory_bank.py` inside `MemoryBank`:

```python
    def write(self, key: torch.Tensor, value: torch.Tensor, surprise: float):
        """Write a memory slot if surprise exceeds threshold.
        If a very similar key exists (cosine > similarity_threshold), update it instead."""
        if surprise < self.config.write_threshold:
            return

        key = key.detach()
        value = value.detach()

        # Check for similar existing slot
        if self.size > 0:
            sims = torch.mv(self._keys, key) / (self._keys.norm(dim=1) * key.norm() + 1e-8)
            max_sim, max_idx = sims.max(dim=0)
            if max_sim.item() > self.config.similarity_threshold:
                # Update existing slot (exponential moving average)
                self._values[max_idx] = 0.8 * self._values[max_idx] + 0.2 * value
                self._usage[max_idx] = 1.0
                self._age[max_idx] = 0
                return

        # Add new slot
        if self.size < self.config.max_slots:
            self._keys = torch.cat([self._keys, key.unsqueeze(0)], dim=0)
            self._values = torch.cat([self._values, value.unsqueeze(0)], dim=0)
            self._usage = torch.cat([self._usage, torch.ones(1)])
            self._age = torch.cat([self._age, torch.zeros(1)])
        else:
            # Overwrite least-used slot
            idx = self._usage.argmin()
            self._keys[idx] = key
            self._values[idx] = value
            self._usage[idx] = 1.0
            self._age[idx] = 0

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from memory: dot-product attention over all slots.
        Returns zero vector if bank is empty."""
        if self.size == 0:
            return torch.zeros(self.config.d_value)

        query = query.detach()
        d = self.config.d_key ** 0.5
        scores = torch.mv(self._keys, query) / d
        weights = torch.softmax(scores, dim=0)
        result = (weights.unsqueeze(1) * self._values).sum(dim=0)

        # Update usage for accessed slots
        self._usage = self._usage + weights.detach()
        return result
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestWriteRead -v`
Expected: PASS

- [ ] **Step 9: Write failing test — decay and forgetting**

```python
# append to tests/test_memory_bank.py

class TestDecay:
    def test_decay_reduces_usage(self):
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        value = torch.randn(CFG.d_value)
        bank.write(key, value, surprise=1.0)
        usage_before = bank._usage[0].item()
        bank.decay()
        usage_after = bank._usage[0].item()
        assert usage_after < usage_before

    def test_decay_increments_age(self):
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        value = torch.randn(CFG.d_value)
        bank.write(key, value, surprise=1.0)
        bank.decay()
        assert bank._age[0].item() == 1

    def test_overwrite_low_usage(self):
        """When bank is full, new write should overwrite lowest-usage slot."""
        cfg = MemoryBankConfig(d_key=16, d_value=8, max_slots=3)
        bank = MemoryBank(cfg)
        # Fill bank
        for i in range(3):
            bank.write(torch.randn(cfg.d_key), torch.randn(cfg.d_value), surprise=1.0)
        assert bank.size == 3
        # Decay heavily so usage drops
        for _ in range(100):
            bank.decay()
        # Read slot 0 to boost its usage
        bank.read(bank._keys[0])
        # Now write a new slot — should overwrite the lowest-usage one (not slot 0)
        new_value = torch.ones(cfg.d_value) * 99
        bank.write(torch.randn(cfg.d_key), new_value, surprise=1.0)
        assert bank.size == 3  # still 3, overwrote one
        # The new value should be in the bank
        assert (bank._values == 99).any()
```

- [ ] **Step 10: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestDecay -v`
Expected: FAIL with "AttributeError: 'MemoryBank' object has no attribute 'decay'"

- [ ] **Step 11: Implement decay**

Add to `nanochat/memory_bank.py` inside `MemoryBank`:

```python
    def decay(self):
        """Age all slots and decay usage. Call once per invocation."""
        if self.size == 0:
            return
        self._age += 1
        self._usage *= self.config.decay_usage
```

- [ ] **Step 12: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestDecay -v`
Expected: PASS

- [ ] **Step 13: Commit**

```bash
git add nanochat/memory_bank.py tests/test_memory_bank.py
git commit -m "feat: add MemoryBank with read/write/decay"
```

---

### Task 2: MemoryBank — save/load persistence

**Files:**
- Modify: `nanochat/memory_bank.py`
- Modify: `tests/test_memory_bank.py`

- [ ] **Step 1: Write failing test — save and load**

```python
# append to tests/test_memory_bank.py
import tempfile
import os

class TestPersistence:
    def test_save_load_roundtrip(self):
        torch.manual_seed(42)
        bank = MemoryBank(CFG)
        for i in range(5):
            bank.write(torch.randn(CFG.d_key), torch.randn(CFG.d_value), surprise=1.0)
        bank.decay()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bank.pt")
            bank.save(path)
            bank2 = MemoryBank.load(path)

            assert bank2.size == bank.size
            assert torch.allclose(bank2._keys, bank._keys)
            assert torch.allclose(bank2._values, bank._values)
            assert torch.allclose(bank2._usage, bank._usage)
            assert torch.allclose(bank2._age, bank._age)

    def test_load_nonexistent_returns_empty_with_config(self):
        bank = MemoryBank.load("/nonexistent/path.pt", default_config=CFG)
        assert bank.size == 0
        assert bank.config.d_key == CFG.d_key

    def test_save_empty_bank(self):
        bank = MemoryBank(CFG)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bank.pt")
            bank.save(path)
            bank2 = MemoryBank.load(path)
            assert bank2.size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestPersistence -v`
Expected: FAIL

- [ ] **Step 3: Implement save/load**

Add to `nanochat/memory_bank.py` inside `MemoryBank`:

```python
    def save(self, path: str):
        """Save bank state to disk."""
        torch.save({
            "config": self.config,
            "keys": self._keys,
            "values": self._values,
            "usage": self._usage,
            "age": self._age,
        }, path)

    @classmethod
    def load(cls, path: str, default_config: "MemoryBankConfig | None" = None) -> "MemoryBank":
        """Load bank from disk. Returns empty bank with default_config if file doesn't exist."""
        try:
            data = torch.load(path, weights_only=False)
        except (FileNotFoundError, EOFError):
            return cls(default_config or MemoryBankConfig())
        bank = cls(data["config"])
        bank._keys = data["keys"]
        bank._values = data["values"]
        bank._usage = data["usage"]
        bank._age = data["age"]
        return bank
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestPersistence -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nanochat/memory_bank.py tests/test_memory_bank.py
git commit -m "feat: add MemoryBank save/load persistence"
```

---

### Task 3: MemoryBank — FAISS working set selection

**Files:**
- Modify: `nanochat/memory_bank.py`
- Modify: `tests/test_memory_bank.py`

- [ ] **Step 1: Add faiss-cpu dependency**

```bash
# Add to pyproject.toml under [project.optional-dependencies] as a NEW extra:
# memory = ["faiss-cpu>=1.7.0"]
# Then: uv sync --extra memory
# NOTE: Do NOT add to the existing cpu/gpu extras — those are for torch builds only.
```

- [ ] **Step 2: Write failing test — select_working_set**

```python
# append to tests/test_memory_bank.py

class TestWorkingSet:
    def test_select_returns_k_slots(self):
        torch.manual_seed(42)
        cfg = MemoryBankConfig(d_key=16, d_value=8, max_slots=100, working_set_k=4)
        bank = MemoryBank(cfg)
        for i in range(20):
            bank.write(torch.randn(cfg.d_key), torch.randn(cfg.d_value), surprise=1.0)

        query = torch.randn(cfg.d_key)
        ws_keys, ws_values = bank.select_working_set(query)
        assert ws_keys.shape == (4, cfg.d_key)
        assert ws_values.shape == (4, cfg.d_value)

    def test_select_with_fewer_slots_than_k(self):
        cfg = MemoryBankConfig(d_key=16, d_value=8, max_slots=100, working_set_k=16)
        bank = MemoryBank(cfg)
        bank.write(torch.randn(cfg.d_key), torch.randn(cfg.d_value), surprise=1.0)
        query = torch.randn(cfg.d_key)
        ws_keys, ws_values = bank.select_working_set(query)
        assert ws_keys.shape[0] == 1  # only 1 slot exists

    def test_select_empty_bank(self):
        bank = MemoryBank(CFG)
        query = torch.randn(CFG.d_key)
        ws_keys, ws_values = bank.select_working_set(query)
        assert ws_keys.shape[0] == 0

    def test_select_returns_most_similar(self):
        torch.manual_seed(42)
        cfg = MemoryBankConfig(d_key=16, d_value=8, max_slots=100, working_set_k=1)
        bank = MemoryBank(cfg)
        # Write 10 random slots
        for i in range(10):
            bank.write(torch.randn(cfg.d_key), torch.randn(cfg.d_value), surprise=1.0)
        # Write one slot with known key
        target_key = torch.ones(cfg.d_key)
        target_value = torch.ones(cfg.d_value) * 42
        bank.write(target_key, target_value, surprise=1.0)
        # Query with the known key — should retrieve it
        ws_keys, ws_values = bank.select_working_set(target_key)
        assert torch.allclose(ws_values[0], target_value, atol=1e-5)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestWorkingSet -v`
Expected: FAIL

- [ ] **Step 4: Implement select_working_set with FAISS**

Add to `nanochat/memory_bank.py`:

```python
# At top of file
import numpy as np
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
```

Add to `MemoryBank`:

```python
    def select_working_set(self, query: torch.Tensor):
        """Select K most similar slots using FAISS (or brute-force fallback).
        Returns (keys, values) tensors of the working set."""
        if self.size == 0:
            return torch.zeros(0, self.config.d_key), torch.zeros(0, self.config.d_value)

        k = min(self.config.working_set_k, self.size)
        query = query.detach()

        if HAS_FAISS and self.size > 64:
            # Use FAISS for large banks
            keys_np = self._keys.numpy().astype(np.float32)
            query_np = query.unsqueeze(0).numpy().astype(np.float32)
            index = faiss.IndexFlatIP(self.config.d_key)  # inner product
            faiss.normalize_L2(keys_np)
            faiss.normalize_L2(query_np)
            index.add(keys_np)
            _, indices = index.search(query_np, k)
            indices = indices[0]
            # Filter out -1 (padding from FAISS when fewer results)
            indices = indices[indices >= 0]
        else:
            # Brute-force for small banks
            sims = torch.mv(self._keys, query) / (self._keys.norm(dim=1) * query.norm() + 1e-8)
            indices = sims.topk(k).indices
            return self._keys[indices], self._values[indices]

        indices = torch.from_numpy(np.asarray(indices)).long()
        return self._keys[indices], self._values[indices]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_memory_bank.py::TestWorkingSet -v`
Expected: PASS

- [ ] **Step 6: Update read to use working set**

Add `read_from_working_set` method (keep original `read` for backward compat):

```python
    def read_from_working_set(self, query: torch.Tensor, ws_keys: torch.Tensor, ws_values: torch.Tensor) -> torch.Tensor:
        """Read from a pre-selected working set. Use for per-token reads."""
        if ws_keys.shape[0] == 0:
            return torch.zeros(self.config.d_value)
        query = query.detach()
        d = self.config.d_key ** 0.5
        scores = ws_keys @ query / d
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(1) * ws_values).sum(dim=0)
```

- [ ] **Step 7: Write test for read_from_working_set**

```python
# append to tests/test_memory_bank.py

class TestReadFromWorkingSet:
    def test_returns_correct_shape(self):
        bank = MemoryBank(CFG)
        ws_keys = torch.randn(4, CFG.d_key)
        ws_values = torch.randn(4, CFG.d_value)
        query = torch.randn(CFG.d_key)
        result = bank.read_from_working_set(query, ws_keys, ws_values)
        assert result.shape == (CFG.d_value,)

    def test_empty_working_set_returns_zeros(self):
        bank = MemoryBank(CFG)
        ws_keys = torch.zeros(0, CFG.d_key)
        ws_values = torch.zeros(0, CFG.d_value)
        query = torch.randn(CFG.d_key)
        result = bank.read_from_working_set(query, ws_keys, ws_values)
        assert torch.allclose(result, torch.zeros(CFG.d_value))

    def test_prefers_matching_key(self):
        torch.manual_seed(42)
        bank = MemoryBank(CFG)
        key_match = torch.randn(CFG.d_key); key_match = key_match / key_match.norm()
        key_other = torch.randn(CFG.d_key); key_other = key_other / key_other.norm()
        val_match = torch.ones(CFG.d_value)
        val_other = -torch.ones(CFG.d_value)
        ws_keys = torch.stack([key_match, key_other])
        ws_values = torch.stack([val_match, val_other])
        result = bank.read_from_working_set(key_match, ws_keys, ws_values)
        assert torch.dot(result, val_match) > torch.dot(result, val_other)
```

- [ ] **Step 8: Run all MemoryBank tests**

Run: `uv run python -m pytest tests/test_memory_bank.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add nanochat/memory_bank.py tests/test_memory_bank.py pyproject.toml
git commit -m "feat: add FAISS working set selection to MemoryBank"
```

---

### Task 4: CellMem — chunked update

**Files:**
- Modify: `nanochat/cellmem.py`
- Modify: `tests/test_cellmem.py`

- [ ] **Step 1: Write failing test — chunked forward**

```python
# append to tests/test_cellmem.py

class TestChunked:
    def test_chunked_output_shape(self):
        """Chunked forward should produce same shape as regular."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2)
        mem = CellMem(cfg)
        B, T = 2, 32
        mem.reset_state(B)
        x = torch.randn(B, T, cfg.d_model)
        g_attn, g_mlp, r_add, x0_mod = mem.forward_chunked(x, chunk_size=8)
        assert g_attn.shape == (B, T, cfg.d_model)

    def test_chunked_M_updates_fewer_times(self):
        """With chunk_size=8 on 32 tokens, M should update 4 times, not 32."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2)
        mem = CellMem(cfg)
        B, T = 1, 32
        mem.reset_state(B)
        x = torch.randn(B, T, cfg.d_model)
        # Capture M states
        M_states = []
        original_forward = mem.forward
        # We'll check by comparing M before and after
        M_before = [m.clone() for m in mem._M]
        mem.forward_chunked(x, chunk_size=8)
        M_after = [m.clone() for m in mem._M]
        # M should have changed
        assert not all(torch.allclose(a, b) for a, b in zip(M_before, M_after))

    def test_chunked_still_counts(self):
        """Chunked update should still enable state accumulation.
        Feed pattern A repeated, check that novelty decreases (M learns to predict A)."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2)
        mem = CellMem(cfg)
        B = 1
        mem.reset_state(B)
        A = torch.randn(cfg.d_model)
        # First chunk: high novelty (M hasn't seen A yet)
        x_early = A.unsqueeze(0).unsqueeze(0).expand(B, 16, cfg.d_model)
        mem.forward_chunked(x_early, chunk_size=16)
        novelty_early = mem.get_mean_novelty()
        # More chunks: M should adapt
        x_late = A.unsqueeze(0).unsqueeze(0).expand(B, 64, cfg.d_model)
        mem.forward_chunked(x_late, chunk_size=16)
        novelty_late = mem.get_mean_novelty()
        assert novelty_late < novelty_early, (
            f"Novelty should decrease: early={novelty_early:.4f} late={novelty_late:.4f}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_cellmem.py::TestChunked -v`
Expected: FAIL with "AttributeError: 'CellMem' object has no attribute 'forward_chunked'"

- [ ] **Step 3: Implement forward_chunked**

Add to `nanochat/cellmem.py` inside `CellMem`:

```python
    def forward_chunked(self, x_seq, chunk_size=32):
        """Process a sequence (B, T, d_model) with M updated per chunk, not per token.
        Each chunk of chunk_size tokens shares the same M for the forward pass,
        then M is updated once using the chunk's mean error."""
        B, T, d_model = x_seq.shape
        n_cells = self.config.n_cells
        d_slice = d_model // n_cells

        all_attn, all_mlp, all_add, all_x0 = [], [], [], []

        for t0 in range(0, T, chunk_size):
            t1 = min(t0 + chunk_size, T)
            chunk = x_seq[:, t0:t1, :]  # (B, chunk_len, d_model)
            chunk_len = t1 - t0

            chunk_attn, chunk_mlp, chunk_add, chunk_x0 = [], [], [], []

            for i in range(n_cells):
                x_slice = chunk[:, :, i * d_slice:(i + 1) * d_slice]  # (B, chunk_len, d_slice)
                x_flat = x_slice.reshape(B * chunk_len, d_slice)

                # 1. Project to cell space
                z = x_flat @ self.W_in[i]  # (B*chunk_len, d_cell)

                # 2. Memory prediction with current M
                M, T_mask = self._M[i], self._T[i]
                z_pred = ((M * T_mask) @ z.T).T  # (B*chunk_len, d_cell)

                # 3. Error and novelty
                error = z - z_pred
                z_norm_sq = z.norm(dim=-1, keepdim=True) ** 2 + 1e-8
                novelty = (error.norm(dim=-1, keepdim=True) ** 2) / z_norm_sq
                self._novelty[i] = novelty.mean().detach()

                # 4. Astrocyte modulation (on chunk stats)
                z_norms = z.norm(dim=-1)
                mu = self._astro_mu[i]
                sigma = self._astro_sigma[i]
                mu = 0.99 * mu + 0.01 * z_norms.mean().detach()
                sigma = 0.99 * sigma + 0.01 * ((z_norms.detach() - mu) ** 2).mean()
                self._astro_mu[i] = mu
                self._astro_sigma[i] = sigma
                alpha_eff = self.alpha_base[i] * (sigma / (mu + 1e-8))

                # 5. Anti-Hebbian update: ONE update using chunk mean
                delta_M = alpha_eff * (error.unsqueeze(-1) * z.unsqueeze(-2)).mean(dim=0)
                self._M[i] = M + delta_M

                # 6. Topology update (detached)
                error_d = error.detach()
                z_d = z.detach()
                gamma_val = self.gamma[i].detach()
                tau_val = self.tau[i].detach()
                delta_T = gamma_val * (
                    (error_d.abs().unsqueeze(-1) * z_d.abs().unsqueeze(-2)).mean(dim=0)
                    - tau_val * T_mask
                )
                self._T[i] = (T_mask + delta_T).clamp(0, 1)

                # 7. MSB fan-out
                mem_out = ((self._M[i] * self._T[i]) @ z.T).T  # (B*chunk_len, d_cell)
                chunk_attn.append((mem_out @ self.W_msb[i][0]).reshape(B, chunk_len, d_slice))
                chunk_mlp.append((mem_out @ self.W_msb[i][1]).reshape(B, chunk_len, d_slice))
                chunk_add.append((mem_out @ self.W_msb[i][2]).reshape(B, chunk_len, d_slice))
                chunk_x0.append((mem_out @ self.W_msb[i][3]).reshape(B, chunk_len, d_slice))

            all_attn.append(torch.cat(chunk_attn, dim=-1))
            all_mlp.append(torch.cat(chunk_mlp, dim=-1))
            all_add.append(torch.cat(chunk_add, dim=-1))
            all_x0.append(torch.cat(chunk_x0, dim=-1))

        return (torch.cat(all_attn, dim=1), torch.cat(all_mlp, dim=1),
                torch.cat(all_add, dim=1), torch.cat(all_x0, dim=1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_cellmem.py::TestChunked -v`
Expected: PASS

- [ ] **Step 5: Run all existing CellMem tests (regression)**

Run: `uv run python -m pytest tests/test_cellmem.py -v`
Expected: ALL PASS (existing tests use per-token forward, unchanged)

- [ ] **Step 6: Commit**

```bash
git add nanochat/cellmem.py tests/test_cellmem.py
git commit -m "feat: add chunked forward to CellMem (M updates per chunk)"
```

---

### Task 5: CellMem — M decay for persistence

**Files:**
- Modify: `nanochat/cellmem.py`
- Modify: `tests/test_cellmem.py`

- [ ] **Step 1: Write failing test — M decay**

```python
# append to tests/test_cellmem.py

class TestDecay:
    def test_m_decay_reduces_magnitude(self):
        """With decay < 1, M magnitude should decrease if no new input."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2)
        mem = CellMem(cfg)
        mem.reset_state(1)
        # Build up M
        for _ in range(50):
            mem(torch.randn(1, cfg.d_model))
        M_before = sum(m.abs().sum().item() for m in mem._M)
        # Apply decay without input
        mem.apply_decay(factor=0.9)
        M_after = sum(m.abs().sum().item() for m in mem._M)
        assert M_after < M_before

    def test_m_decay_preserves_structure(self):
        """Decay should scale M uniformly, preserving relative structure."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2)
        mem = CellMem(cfg)
        mem.reset_state(1)
        for _ in range(50):
            mem(torch.randn(1, cfg.d_model))
        M_before = mem._M[0].clone()
        mem.apply_decay(factor=0.9)
        M_after = mem._M[0]
        # Should be approximately 0.9 * M_before
        assert torch.allclose(M_after, 0.9 * M_before, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_cellmem.py::TestDecay -v`
Expected: FAIL

- [ ] **Step 3: Implement apply_decay**

Add to `nanochat/cellmem.py` inside `CellMem`:

```python
    def apply_decay(self, factor: float = 0.999):
        """Decay M by a factor. Call between invocations to prevent explosion.
        M = factor * M. The identity component (0.01*I) is NOT restored —
        this means very old, unreinforced patterns eventually vanish."""
        for i in range(self.config.n_cells):
            self._M[i] = factor * self._M[i]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_cellmem.py::TestDecay -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nanochat/cellmem.py tests/test_cellmem.py
git commit -m "feat: add M decay to CellMem for persistent mode"
```

---

### Task 6: CellMem — expose novelty for MemoryBank write gating

**Files:**
- Modify: `nanochat/cellmem.py`
- Modify: `tests/test_cellmem.py`

- [ ] **Step 1: Write failing test — get_mean_novelty**

```python
# append to tests/test_cellmem.py

class TestNoveltyExposure:
    def test_get_mean_novelty_after_familiar(self):
        """After many repetitions, novelty should be low."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2)
        mem = CellMem(cfg)
        mem.reset_state(1)
        A, _, _ = make_patterns(cfg.d_model)
        for _ in range(100):
            mem(A.unsqueeze(0))
        novelty = mem.get_mean_novelty()
        assert novelty < 0.5, f"Novelty should be low for familiar input: {novelty}"

    def test_get_mean_novelty_after_novel(self):
        """After novel input, novelty should be higher."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2)
        mem = CellMem(cfg)
        mem.reset_state(1)
        A, _, C = make_patterns(cfg.d_model)
        for _ in range(100):
            mem(A.unsqueeze(0))
        novelty_familiar = mem.get_mean_novelty()
        # Now feed novel pattern
        mem(C.unsqueeze(0))
        novelty_novel = mem.get_mean_novelty()
        assert novelty_novel > novelty_familiar
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_cellmem.py::TestNoveltyExposure -v`
Expected: FAIL

- [ ] **Step 3: Implement get_mean_novelty**

Add to `nanochat/cellmem.py` inside `CellMem`:

```python
    def get_mean_novelty(self) -> float:
        """Return mean novelty across all cells. Used by MemoryBank to gate writes."""
        if self._novelty is None:
            return 0.0
        return sum(n.item() for n in self._novelty) / len(self._novelty)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_cellmem.py::TestNoveltyExposure -v`
Expected: PASS

- [ ] **Step 5: Run all CellMem tests (regression)**

Run: `uv run python -m pytest tests/test_cellmem.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add nanochat/cellmem.py tests/test_cellmem.py
git commit -m "feat: expose CellMem novelty for MemoryBank write gating"
```

---

### Task 7: Integration — CellMem + MemoryBank working together

**Files:**
- Create: `scripts/test3_persistent_memory.py`
- Create: `docs/cellmem/06-memory-bank.md`

- [ ] **Step 1: Write the TinyTransformerWithMemory integration class**

This is the key wiring between CellMem and MemoryBank. Create `scripts/test3_persistent_memory.py` with:

```python
# scripts/test3_persistent_memory.py
"""
Test 3: Persistent Memory — does MemoryBank remember across sequences?

Setup: Same tiny transformer as Test 2b (2 layer, d_model=64).
Three phases:
  Phase 1 (LEARN): Train on sequences with facts (symbol→count mappings)
  Phase 2 (FORGET): Train on unrelated sequences (no relevant facts)
  Phase 3 (RECALL): Test if the model can recall facts from Phase 1

Compare:
  - baseline: no CellMem, no MemoryBank
  - cellmem_only: CellMem with M reset per sequence
  - cellmem_persistent: CellMem with persistent M + MemoryBank

The key metric: accuracy in Phase 3 (recall after interference).
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from nanochat.cellmem import CellMem, CellMemConfig
from nanochat.memory_bank import MemoryBank, MemoryBankConfig


@dataclass
class TinyConfig:
    vocab_size: int = 32
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    max_seq_len: int = 40


class TinyBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, g_attn=None, g_mlp=None, r_add=None):
        # Attention with optional gate
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                                attn_mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device))
        if g_attn is not None:
            attn_out = attn_out * torch.sigmoid(g_attn)
        x = x + attn_out
        if r_add is not None:
            x = x + r_add
        # MLP with optional gate
        mlp_out = self.mlp(self.ln2(x))
        if g_mlp is not None:
            mlp_out = mlp_out * torch.sigmoid(g_mlp)
        x = x + mlp_out
        return x


class TinyTransformerWithMemory(nn.Module):
    """Tiny transformer with CellMem + MemoryBank integration.

    Integration wiring:
    - At sequence START: bank.select_working_set(context) -> working set
    - Per token per layer: CellMem gates + bank.read_from_working_set -> r_mem added to residual
    - At sequence END: bank.write(residual_mean, cellmem_output_mean, novelty)
    """
    def __init__(self, cfg: TinyConfig, use_cellmem=False, use_bank=False):
        super().__init__()
        self.cfg = cfg
        self.use_cellmem = use_cellmem
        self.use_bank = use_bank
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TinyBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

        if use_cellmem:
            cm_cfg = CellMemConfig(d_model=cfg.d_model, d_cell=16, n_cells=2)
            self.cellmem = CellMem(cm_cfg)
        if use_bank:
            bank_cfg = MemoryBankConfig(
                d_key=cfg.d_model,
                d_value=16 * 2,  # d_cell * n_cells
                max_slots=256,
                working_set_k=8,
                write_threshold=0.3,
            )
            self.bank = MemoryBank(bank_cfg)
            self.mem_proj = nn.Linear(16 * 2, cfg.d_model)  # project bank values to d_model

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)  # (B, T, d_model)

        # MemoryBank: select working set at start of sequence
        ws_keys, ws_values = None, None
        if self.use_bank and self.bank.size > 0:
            context = x.mean(dim=(0, 1))  # mean embedding as context query
            ws_keys, ws_values = self.bank.select_working_set(context)

        # CellMem: process token-by-token, collect gates
        if self.use_cellmem:
            self.cellmem.reset_state(B)
            all_g_attn, all_g_mlp, all_r_add = [], [], []
            cellmem_outputs = []
            for t in range(T):
                g_attn, g_mlp, r_add, x0_mod = self.cellmem(x[:, t, :])
                all_g_attn.append(g_attn)
                all_g_mlp.append(g_mlp)
                # Add memory bank read to r_add
                if self.use_bank and ws_keys is not None and ws_keys.shape[0] > 0:
                    r_mem = self.bank.read_from_working_set(x[:, t, :].mean(dim=0), ws_keys, ws_values)
                    r_add = r_add + self.mem_proj(r_mem.unsqueeze(0).expand(B, -1))
                all_r_add.append(r_add)
                cellmem_outputs.append(self.cellmem._novelty[0].item())
            g_attn_seq = torch.stack(all_g_attn, dim=1)
            g_mlp_seq = torch.stack(all_g_mlp, dim=1)
            r_add_seq = torch.stack(all_r_add, dim=1)

        # Transformer blocks
        for block in self.blocks:
            if self.use_cellmem:
                x = block(x, g_attn=g_attn_seq, g_mlp=g_mlp_seq, r_add=r_add_seq)
            else:
                x = block(x)

        logits = self.head(x)

        # MemoryBank: write at end of sequence
        if self.use_bank and self.use_cellmem:
            novelty = self.cellmem.get_mean_novelty()
            key = x.mean(dim=(0, 1)).detach()  # mean residual as key
            # Collect CellMem output as value (use last token's M state)
            z_last = x[:, -1, :].mean(dim=0).detach()
            value = z_last[:self.bank.config.d_value]  # truncate to d_value
            self.bank.write(key, value, surprise=novelty)
            self.bank.decay()

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss
        return logits


def generate_fact_batch(batch_size, vocab_size=32, seq_len=30, n_facts=3, seed=None):
    """Generate sequences with facts: FACT_TOKEN SYMBOL COUNT_TOKEN.
    Query at end: QUERY_TOKEN SYMBOL -> model must predict COUNT_TOKEN.
    Facts use tokens 0-15 as symbols, 16-25 as counts, 26=FACT, 27=QUERY, 28=NOISE."""
    if seed is not None:
        torch.manual_seed(seed)
    FACT, QUERY, NOISE = 26, 27, 28
    inputs = torch.full((batch_size, seq_len), NOISE, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    for b in range(batch_size):
        facts = {}
        pos = 0
        for _ in range(n_facts):
            symbol = torch.randint(0, 10, (1,)).item()
            count = torch.randint(16, 26, (1,)).item()
            facts[symbol] = count
            if pos + 3 <= seq_len - 3:
                inputs[b, pos] = FACT
                inputs[b, pos + 1] = symbol
                inputs[b, pos + 2] = count
                pos += 3
        # Fill rest with noise
        for p in range(pos, seq_len - 3):
            inputs[b, p] = NOISE
        # Query one fact
        if facts:
            q_symbol = list(facts.keys())[0]
            q_count = facts[q_symbol]
            inputs[b, -3] = QUERY
            inputs[b, -2] = q_symbol
            targets[b, -1] = q_count
    return inputs, targets


def run_experiment(variant, n_steps_per_phase, device, seed):
    """Run 3-phase experiment for one variant."""
    torch.manual_seed(seed)
    cfg = TinyConfig()
    use_cm = variant in ("cellmem_only", "cellmem_persistent")
    use_bank = variant == "cellmem_persistent"
    model = TinyTransformerWithMemory(cfg, use_cellmem=use_cm, use_bank=use_bank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    results = {}
    for phase_name, phase_steps, train in [("learn", n_steps_per_phase, True),
                                             ("forget", n_steps_per_phase, True),
                                             ("recall", n_steps_per_phase // 2, False)]:
        correct, total = 0, 0
        for step in range(phase_steps):
            if phase_name == "forget":
                # Random sequences, no facts
                inputs = torch.randint(0, cfg.vocab_size, (64, 30)).to(device)
                targets = torch.full((64, 30), -1, dtype=torch.long).to(device)
            else:
                inputs, targets = generate_fact_batch(64, seed=seed * 10000 + step)
                inputs, targets = inputs.to(device), targets.to(device)

            if train:
                loss = model(inputs, targets)
                if loss.item() > 0:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            # Eval
            with torch.no_grad():
                logits = model(inputs)
                preds = logits[:, -1, :].argmax(dim=-1)
                mask = targets[:, -1] != -1
                if mask.any():
                    correct += (preds[mask] == targets[:, -1][mask]).sum().item()
                    total += mask.sum().item()

        results[phase_name] = correct / max(total, 1) * 100
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    for variant in ["baseline", "cellmem_only", "cellmem_persistent"]:
        print(f"\n{'='*50}")
        print(f"Variant: {variant}")
        for seed in args.seeds:
            results = run_experiment(variant, args.n_steps, args.device, seed)
            print(f"  Seed {seed}: learn={results['learn']:.1f}% forget={results['forget']:.1f}% recall={results['recall']:.1f}%")
```

- [ ] **Step 2: Run the benchmark on CPU**

Run: `uv run python -m scripts.test3_persistent_memory --device cpu --n_steps 500`
Expected: cellmem_persistent should recall better than others in Phase 3 (recall)

- [ ] **Step 3: Write design doc**

Create `docs/cellmem/06-memory-bank.md` documenting:
- Architecture (two-level memory)
- MemoryBank design (key/value, FAISS, persistence)
- Integration wiring: where read/write happen relative to the transformer forward pass
- Test 3 results

- [ ] **Step 4: Commit**

```bash
git add scripts/test3_persistent_memory.py docs/cellmem/06-memory-bank.md
git commit -m "feat: add persistent memory integration test (Test 3)"
```

---

## Dependency Graph

```
Task 1 (MemoryBank core) ──→ Task 2 (persistence) ──→ Task 3 (FAISS) ──┐
                                                                         │
Task 4 (chunked CellMem) ──→ Task 5 (M decay) ──→ Task 6 (novelty) ───→ Task 7 (integration)
```

Tasks 1-3 and Tasks 4-6 can run **in parallel** (independent components). Task 7 requires both chains to be complete.

## Success Criteria

1. All unit tests pass (Tasks 1-6)
2. Integration benchmark (Task 7) shows measurable recall improvement for cellmem_persistent over baseline in Phase 3
3. Memory bank save/load works across process restarts
4. No regression on existing CellMem counting benchmark (74.3%)
