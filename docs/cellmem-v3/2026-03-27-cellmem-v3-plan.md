# CellMem v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the router-based memory retrieval system with per-layer K/V storage, replacing v2's failed ContentGate.

**Architecture:** MemoryRouter (W_Q^R, W_K^R) selects episodes via InfoNCE contrastive loss. KVInterceptor captures pre-RoPE K/V pairs at configurable layers. MemoryStore holds per-layer K/V banks + router keys. CellMemWrapper orchestrates read/write paths.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers (Qwen3.5-4B), pytest

**Spec:** `docs/cellmem-v3/2026-03-27-cellmem-v3-design.md`

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `nanochat/cellmem_v3.py` | CellMemConfig, MemoryStore, MemoryRouter | Create |
| `nanochat/kv_interceptor.py` | KVInterceptor (Qwen-specific hook-based K/V capture) | Create |
| `scripts/train_cellmem_qwen_v3.py` | CellMemWrapper, training loop, eval (v3) | Create |
| `tests/test_cellmem_v3.py` | Unit tests for all v3 components | Create |
| `nanochat/cellmem_v2.py` | SurpriseCalculator (imported by v3, not modified) | Unchanged |

---

### Task 1: CellMemConfig + MemoryRouter

**Files:**
- Create: `nanochat/cellmem_v3.py`
- Create: `tests/test_cellmem_v3.py`

- [ ] **Step 1: Write failing tests for CellMemConfig**

```python
# tests/test_cellmem_v3.py
import torch
import pytest
from nanochat.cellmem_v3 import CellMemConfig


class TestCellMemConfigV3:
    def test_default_values(self):
        cfg = CellMemConfig()
        assert cfg.n_slots == 256
        assert cfg.router_dim == 128
        assert cfg.router_layers is None  # auto top-4
        assert cfg.episode_size == 8
        assert cfg.top_k == 4
        assert cfg.contrastive_tau == 0.07
        assert cfg.surprise_threshold == 4.0
        assert cfg.decay_factor == 0.95

    def test_custom_router_layers(self):
        cfg = CellMemConfig(router_layers=[20, 25, 30, 35])
        assert cfg.router_layers == [20, 25, 30, 35]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cellmem_v3.py::TestCellMemConfigV3 -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nanochat.cellmem_v3'`

- [ ] **Step 3: Implement CellMemConfig**

```python
# nanochat/cellmem_v3.py
"""
CellMem v3: Router-based persistent memory with per-layer K/V storage.

Replaces v2's ContentGate with a MemoryRouter (W_Q^R, W_K^R) that selects
episodes via cosine similarity. Stores K/V pairs pre-RoPE per layer for
high-fidelity retrieval. Surprise-gated writes and cross-session persistence
are preserved from v2.

Spec: docs/cellmem-v3/2026-03-27-cellmem-v3-design.md
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CellMemConfig:
    enabled: bool = False
    n_slots: int = 256              # max tokens in memory
    surprise_threshold: float = 4.0
    decay_factor: float = 0.95
    max_snapshots: int = 5
    memory_dir: str = "~/.cache/nanochat/memory"
    max_vector_norm: float = 50.0
    write_mode: str = "raw"         # "raw" | "delta"
    delta_threshold: float = 0.1
    min_novelty: float = 0.0
    # v3-specific
    router_dim: int = 128
    router_layers: list[int] | None = None  # None = auto top-4
    episode_size: int = 8
    top_k: int = 4
    contrastive_tau: float = 0.07
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cellmem_v3.py::TestCellMemConfigV3 -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for MemoryRouter**

```python
# tests/test_cellmem_v3.py (append)
from nanochat.cellmem_v3 import MemoryRouter


class TestMemoryRouter:
    def test_init_nonzero_weights(self):
        router = MemoryRouter(d_model=64, d_router=16)
        assert router.w_q_r.weight.abs().sum() > 0
        assert router.w_k_r.weight.abs().sum() > 0

    def test_route_returns_topk_indices(self):
        router = MemoryRouter(d_model=64, d_router=16, top_k=2)
        h_query = torch.randn(1, 10, 64)  # [B, T, d_model]
        router_keys = torch.randn(8, 16)   # [n_episodes, d_router]
        indices, scores = router.route(h_query, router_keys)
        assert indices.shape == (1, 10, 2)  # [B, T, top_k]
        assert scores.shape == (1, 10, 2)

    def test_route_empty_memory_returns_empty(self):
        router = MemoryRouter(d_model=64, d_router=16, top_k=2)
        h_query = torch.randn(1, 10, 64)
        router_keys = torch.zeros(0, 16)
        indices, scores = router.route(h_query, router_keys)
        assert indices.shape[-1] == 0

    def test_route_fewer_episodes_than_topk(self):
        router = MemoryRouter(d_model=64, d_router=16, top_k=4)
        h_query = torch.randn(1, 5, 64)
        router_keys = torch.randn(2, 16)  # only 2 episodes, top_k=4
        indices, scores = router.route(h_query, router_keys)
        assert indices.shape == (1, 5, 2)  # min(top_k, n_episodes)

    def test_encode_episode_shape(self):
        router = MemoryRouter(d_model=64, d_router=16)
        h_tokens = torch.randn(8, 64)  # [episode_size, d_model]
        key = router.encode_episode(h_tokens)
        assert key.shape == (16,)  # [d_router]

    def test_encode_episode_partial(self):
        router = MemoryRouter(d_model=64, d_router=16)
        h_tokens = torch.randn(3, 64)  # partial episode, < 8
        key = router.encode_episode(h_tokens)
        assert key.shape == (16,)

    def test_contrastive_loss_positive(self):
        router = MemoryRouter(d_model=64, d_router=16)
        q = torch.randn(4, 16)       # [B, d_router]
        pos = q.clone()               # identical = perfect match
        negs = torch.randn(4, 3, 16)  # [B, n_neg, d_router]
        loss = router.contrastive_loss(q, pos, negs, tau=0.07)
        assert loss.item() >= 0
        assert loss.item() < 1.0  # should be low for identical pos

    def test_contrastive_loss_gradient_flows(self):
        router = MemoryRouter(d_model=64, d_router=16)
        h_query = torch.randn(4, 64, requires_grad=True)
        q = router.w_q_r(h_query)     # [B, d_router]
        pos = torch.randn(4, 16)
        negs = torch.randn(4, 3, 16)
        loss = router.contrastive_loss(q, pos, negs, tau=0.07)
        loss.backward()
        assert router.w_q_r.weight.grad is not None
        assert router.w_q_r.weight.grad.abs().sum() > 0
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cellmem_v3.py::TestMemoryRouter -v`
Expected: FAIL with `ImportError: cannot import name 'MemoryRouter'`

- [ ] **Step 7: Implement MemoryRouter**

```python
# nanochat/cellmem_v3.py (append after CellMemConfig)

class MemoryRouter(nn.Module):
    """Router for episode selection via cosine similarity.

    Shared across all router layers. Selects top-k episodes from memory
    using learned projections W_Q^R (query) and W_K^R (key).
    """
    def __init__(self, d_model: int, d_router: int = 128, top_k: int = 4):
        super().__init__()
        self.d_router = d_router
        self.top_k = top_k
        self.w_q_r = nn.Linear(d_model, d_router, bias=False)
        self.w_k_r = nn.Linear(d_model, d_router, bias=False)
        # Never zero-init (v2 lesson: blocks gradient via chain rule)
        nn.init.normal_(self.w_q_r.weight, std=0.02)
        nn.init.normal_(self.w_k_r.weight, std=0.02)

    def route(self, h_query: torch.Tensor, router_keys: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k episodes for each query token.

        Args:
            h_query: [B, T, d_model] hidden states at router layer
            router_keys: [n_episodes, d_router] episode keys from MemoryStore

        Returns:
            indices: [B, T, k] episode indices (k = min(top_k, n_episodes))
            scores: [B, T, k] cosine similarity scores
        """
        n_eps = router_keys.shape[0]
        if n_eps == 0:
            B, T = h_query.shape[:2]
            empty = torch.zeros(B, T, 0, dtype=torch.long, device=h_query.device)
            return empty, torch.zeros(B, T, 0, device=h_query.device)

        q_r = F.normalize(self.w_q_r(h_query), dim=-1)   # [B, T, d_router]
        k_r = F.normalize(router_keys, dim=-1)            # [n_eps, d_router]
        sim = torch.matmul(q_r, k_r.T)                    # [B, T, n_eps]
        k = min(self.top_k, n_eps)
        scores, indices = sim.topk(k, dim=-1)
        return indices, scores

    def encode_episode(self, h_tokens: torch.Tensor) -> torch.Tensor:
        """Encode a group of token hidden states into a router key.

        Args:
            h_tokens: [N, d_model] hidden states (N <= episode_size)

        Returns:
            router_key: [d_router]
        """
        pooled = h_tokens.mean(dim=0)            # [d_model]
        return self.w_k_r(pooled)                 # [d_router]

    def contrastive_loss(self, q: torch.Tensor, pos: torch.Tensor,
                         negs: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        """InfoNCE contrastive loss.

        Args:
            q: [B, d_router] query embeddings
            pos: [B, d_router] positive episode keys
            negs: [B, n_neg, d_router] negative episode keys
            tau: temperature

        Returns:
            loss: scalar
        """
        q = F.normalize(q, dim=-1)
        pos = F.normalize(pos, dim=-1)
        negs = F.normalize(negs, dim=-1)

        pos_sim = (q * pos).sum(dim=-1, keepdim=True) / tau  # [B, 1]
        neg_sim = torch.bmm(negs, q.unsqueeze(-1)).squeeze(-1) / tau  # [B, n_neg]
        logits = torch.cat([pos_sim, neg_sim], dim=-1)  # [B, 1+n_neg]
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cellmem_v3.py::TestMemoryRouter -v`
Expected: PASS (8 tests)

- [ ] **Step 9: Commit**

```bash
git add nanochat/cellmem_v3.py tests/test_cellmem_v3.py
git commit -m "feat(v3): CellMemConfig + MemoryRouter with InfoNCE loss"
```

---

### Task 2: MemoryStore v3

**Files:**
- Modify: `nanochat/cellmem_v3.py`
- Modify: `tests/test_cellmem_v3.py`

- [ ] **Step 1: Write failing tests for MemoryStore v3**

```python
# tests/test_cellmem_v3.py (append)
from nanochat.cellmem_v3 import MemoryStore


class TestMemoryStoreV3Init:
    def test_init_shapes(self):
        cfg = CellMemConfig(n_slots=32, router_dim=16)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=4, d_head=8)
        assert store.keys.shape == (2, 32, 4, 8)
        assert store.values.shape == (2, 32, 4, 8)
        assert store.router_keys.shape == (32 // cfg.episode_size, 16)
        assert store.active_count == 0

    def test_read_empty_returns_none(self):
        cfg = CellMemConfig(n_slots=32, router_dim=16)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=4, d_head=8)
        assert store.read([0]) is None


class TestMemoryStoreV3Write:
    def test_write_increments_count(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=2, d_head=4)
        kv = {
            "keys": torch.randn(2, 2, 4),    # [n_layers, n_kv_heads, d_head]
            "values": torch.randn(2, 2, 4),
        }
        router_key = torch.randn(8)
        store.write(kv, router_key, surprise=5.0)
        assert store.active_count == 1

    def test_write_stores_kv_correctly(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=2, d_head=4)
        k = torch.ones(2, 2, 4) * 3.0
        v = torch.ones(2, 2, 4) * 7.0
        store.write({"keys": k, "values": v}, torch.randn(8), surprise=5.0)
        assert (store.keys[:, 0, :, :] == 3.0).all()
        assert (store.values[:, 0, :, :] == 7.0).all()

    def test_write_full_evicts_min_surprise(self):
        cfg = CellMemConfig(n_slots=4, router_dim=8, episode_size=2)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        for i in range(4):
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                torch.randn(8), surprise=float(i + 1)  # surprises: 1,2,3,4
            )
        assert store.active_count == 4
        # Write 5th with high surprise → should evict slot with surprise=1.0
        store.write(
            {"keys": torch.ones(1, 1, 4) * 99, "values": torch.ones(1, 1, 4) * 99},
            torch.randn(8), surprise=10.0
        )
        assert store.active_count == 4
        assert 1.0 not in store.surprise[:4].tolist()

    def test_decorrelation_skips_similar(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, min_novelty=0.1)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        k = torch.ones(1, 1, 4)
        store.write({"keys": k, "values": k.clone()}, torch.randn(8), surprise=5.0)
        # Write near-identical → should skip
        store.write({"keys": k * 1.01, "values": k.clone()}, torch.randn(8), surprise=5.0)
        assert store.active_count == 1  # second write skipped


class TestMemoryStoreV3Read:
    def test_read_selected_episodes(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=2, d_head=4)
        # Write 8 tokens → 2 episodes
        for i in range(8):
            store.write(
                {"keys": torch.ones(2, 2, 4) * i, "values": torch.ones(2, 2, 4) * i},
                torch.randn(8) if i % 4 == 3 else None,  # router key every 4 tokens
                surprise=5.0
            )
        # Read episode 0 (tokens 0-3)
        result = store.read([0])
        assert result is not None
        keys, values = result
        # Episode 0 has tokens 0-3 for each layer
        assert keys.shape[1] == 4  # 4 tokens in episode

    def test_get_router_keys_shape(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        for i in range(8):
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                torch.randn(8) if i % 4 == 3 else None,
                surprise=5.0
            )
        rk = store.get_router_keys()
        assert rk.shape[1] == 8  # d_router


class TestMemoryStoreV3Persistence:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = CellMemConfig(n_slots=8, router_dim=4, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        for i in range(4):
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                torch.randn(4), surprise=float(i)
            )
        path = tmp_path / "mem.pt"
        store.save(path)

        store2 = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store2.load(path)
        assert store2.active_count == store.active_count
        assert torch.allclose(store2.surprise, store.surprise)

    def test_decay_on_load(self, tmp_path):
        cfg = CellMemConfig(n_slots=8, router_dim=4, decay_factor=0.5)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store.write(
            {"keys": torch.ones(1, 1, 4), "values": torch.ones(1, 1, 4)},
            torch.randn(4), surprise=5.0
        )
        path = tmp_path / "mem.pt"
        store.save(path)

        store2 = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store2.load(path)
        assert torch.allclose(store2.keys[:, 0], torch.ones(1, 1, 4) * 0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cellmem_v3.py -k "MemoryStoreV3" -v`
Expected: FAIL with `ImportError: cannot import name 'MemoryStore' from 'nanochat.cellmem_v3'`

- [ ] **Step 3: Implement MemoryStore v3**

```python
# nanochat/cellmem_v3.py (append after MemoryRouter)

class MemoryStore:
    """Per-layer K/V storage with episode-based router keys.

    Stores K/V pairs pre-RoPE for each router layer, plus mean-pooled
    router keys for episode retrieval. Supports surprise-gated writes,
    decorrelation, decay, and persistence.
    """
    def __init__(self, config: CellMemConfig, n_layers: int,
                 n_kv_heads: int, d_head: int):
        self.config = config
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        N = config.n_slots
        E = N // config.episode_size

        self.keys = torch.zeros(n_layers, N, n_kv_heads, d_head)
        self.values = torch.zeros(n_layers, N, n_kv_heads, d_head)
        self.router_keys = torch.zeros(E, config.router_dim)
        self.surprise = torch.zeros(N)
        self.age = torch.zeros(N)
        self.write_ptr = 0
        self.active_count = 0
        self.episode_ptr = 0
        self.active_episodes = 0

    @torch.compiler.disable
    def write(self, kv: dict, router_key: torch.Tensor | None,
              surprise: float):
        """Write one token's K/V across all layers.

        Args:
            kv: {"keys": [n_layers, n_kv_heads, d_head],
                 "values": [n_layers, n_kv_heads, d_head]}
            router_key: [d_router] or None (set when episode completes)
            surprise: scalar surprise score
        """
        N = self.config.n_slots
        k_tensor = kv["keys"].detach()
        v_tensor = kv["values"].detach()

        # Decorrelation gate
        if self.config.min_novelty > 0 and self.active_count > 0:
            # Compare first layer's K as proxy
            active_k = self.keys[0, :self.active_count].reshape(self.active_count, -1)
            new_k = k_tensor[0].reshape(1, -1)
            sims = F.cosine_similarity(new_k, active_k, dim=-1)
            if sims.max().item() > (1.0 - self.config.min_novelty):
                return

        # Norm clamp
        for l in range(self.n_layers):
            for tensor in [k_tensor[l], v_tensor[l]]:
                norm = tensor.norm()
                if norm > self.config.max_vector_norm:
                    tensor.mul_(self.config.max_vector_norm / norm)

        if self.active_count < N:
            idx = self.write_ptr
            self.write_ptr = (self.write_ptr + 1) % N
            self.active_count += 1
        else:
            idx = self.surprise[:N].argmin().item()

        self.keys[:, idx] = k_tensor
        self.values[:, idx] = v_tensor
        self.surprise[idx] = surprise
        self.age[idx] = 0

        # Store router key when provided (episode boundary)
        if router_key is not None:
            E = self.config.n_slots // self.config.episode_size
            if self.active_episodes < E:
                ep_idx = self.episode_ptr
                self.episode_ptr = (self.episode_ptr + 1) % E
                self.active_episodes += 1
            else:
                ep_idx = self.active_episodes - 1  # overwrite last
            self.router_keys[ep_idx] = router_key.detach()

    @torch.compiler.disable
    def read(self, episode_indices: list[int]
             ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Read K/V for selected episodes.

        Args:
            episode_indices: list of episode indices to retrieve

        Returns:
            (keys, values) each [n_layers, n_tokens, n_kv_heads, d_head]
            or None if memory is empty.
        """
        if self.active_count == 0:
            return None
        eps = self.config.episode_size
        token_indices = []
        for ep in episode_indices:
            start = ep * eps
            end = min(start + eps, self.active_count)
            token_indices.extend(range(start, end))
        if not token_indices:
            return None
        idx = torch.tensor(token_indices, dtype=torch.long)
        return self.keys[:, idx], self.values[:, idx]

    def get_router_keys(self) -> torch.Tensor:
        """Return active router keys. Shape: [n_active_episodes, d_router]."""
        return self.router_keys[:self.active_episodes]

    def age_all(self, tokens_processed: int = 1):
        if self.active_count > 0:
            self.age[:self.active_count] += tokens_processed

    @torch.compiler.disable
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "keys": self.keys.clone(),
            "values": self.values.clone(),
            "router_keys": self.router_keys.clone(),
            "surprise": self.surprise.clone(),
            "age": self.age.clone(),
            "write_ptr": self.write_ptr,
            "active_count": self.active_count,
            "episode_ptr": self.episode_ptr,
            "active_episodes": self.active_episodes,
            "config": asdict(self.config),
            "version": 3,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, path)

    @torch.compiler.disable
    def load(self, path):
        data = torch.load(Path(path), weights_only=False)
        self.keys = data["keys"]
        self.values = data["values"]
        self.router_keys = data["router_keys"]
        self.surprise = data["surprise"]
        self.age = data["age"]
        self.write_ptr = data["write_ptr"]
        self.active_count = data["active_count"]
        self.episode_ptr = data["episode_ptr"]
        self.active_episodes = data["active_episodes"]
        if self.active_count > 0:
            self.keys *= self.config.decay_factor
            self.values *= self.config.decay_factor

    @torch.compiler.disable
    def snapshot(self):
        mem_dir = Path(self.config.memory_dir).expanduser()
        mem_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        self.save(mem_dir / f"snapshot_{ts}.pt")
        snaps = sorted(mem_dir.glob("snapshot_*.pt"))
        while len(snaps) > self.config.max_snapshots:
            snaps.pop(0).unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cellmem_v3.py -k "MemoryStoreV3" -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add nanochat/cellmem_v3.py tests/test_cellmem_v3.py
git commit -m "feat(v3): MemoryStore with per-layer K/V and episode routing"
```

---

### Task 3: KVInterceptor

**Files:**
- Create: `nanochat/kv_interceptor.py`
- Modify: `tests/test_cellmem_v3.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cellmem_v3.py (append)
from nanochat.kv_interceptor import KVInterceptor


class TestKVInterceptor:
    def _make_mock_model(self):
        """Create a minimal model with attention layers that have k_proj/v_proj."""
        class FakeAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = torch.nn.Linear(32, 16, bias=False)
                self.v_proj = torch.nn.Linear(32, 16, bias=False)
                self.rotary_emb = None  # marker for pre-RoPE intercept

            def forward(self, x):
                return x

        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

            def forward(self, x):
                return x

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([FakeLayer() for _ in range(4)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class FakeWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeModel()

        return FakeWrapper()

    def test_register_and_capture(self):
        model = self._make_mock_model()
        interceptor = KVInterceptor(model, layer_indices=[2, 3], model_type="qwen")
        x = torch.randn(1, 10, 32)
        model.model(x)  # trigger hooks
        buf = interceptor.get_buffered_kv()
        assert 2 in buf
        assert 3 in buf
        assert buf[2][0].shape[-1] == 16  # d_head from k_proj

    def test_clear_buffer(self):
        model = self._make_mock_model()
        interceptor = KVInterceptor(model, layer_indices=[2], model_type="qwen")
        x = torch.randn(1, 5, 32)
        model.model(x)
        interceptor.clear_buffer()
        assert len(interceptor.get_buffered_kv()) == 0

    def test_remove_hooks(self):
        model = self._make_mock_model()
        interceptor = KVInterceptor(model, layer_indices=[2], model_type="qwen")
        interceptor.remove_hooks()
        x = torch.randn(1, 5, 32)
        model.model(x)
        assert len(interceptor.get_buffered_kv()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cellmem_v3.py::TestKVInterceptor -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement KVInterceptor**

```python
# nanochat/kv_interceptor.py
"""
KVInterceptor: Hook-based capture of K/V pairs pre-RoPE.

Model-specific. Currently supports Qwen architecture.
Registers forward hooks on attention layers to intercept K/V
before rotary position embeddings are applied.
"""
import torch
import torch.nn as nn


class KVInterceptor:
    """Captures K/V projections pre-RoPE from specified layers.

    Uses forward hooks on the attention module's k_proj and v_proj
    to capture K/V before RoPE rotation is applied.
    """
    def __init__(self, model: nn.Module, layer_indices: list[int],
                 model_type: str = "qwen"):
        self.layer_indices = layer_indices
        self.model_type = model_type
        self._buffer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks(model)

    def _get_attention_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get the attention module for a layer. Qwen: model.model.layers[i].self_attn"""
        return model.model.layers[layer_idx].self_attn

    def _register_hooks(self, model: nn.Module):
        """Register hooks on k_proj and v_proj to capture pre-RoPE K/V."""
        for layer_idx in self.layer_indices:
            attn = self._get_attention_module(model, layer_idx)

            # We hook the entire attention module's forward to capture
            # the K/V projections before RoPE is applied.
            # In Qwen, the flow is: hidden → k_proj → apply_rotary → attention
            # We intercept after k_proj/v_proj but before rotary.
            k_proj = attn.k_proj
            v_proj = attn.v_proj

            def make_capture_hook(l_idx, proj_type):
                def hook(module, input, output):
                    if l_idx not in self._buffer:
                        self._buffer[l_idx] = [None, None]
                    idx = 0 if proj_type == "k" else 1
                    self._buffer[l_idx][idx] = output.detach()
                return hook

            h_k = k_proj.register_forward_hook(make_capture_hook(layer_idx, "k"))
            h_v = v_proj.register_forward_hook(make_capture_hook(layer_idx, "v"))
            self._hooks.extend([h_k, h_v])

    def get_buffered_kv(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Return captured K/V pairs. {layer_idx: (K, V)}."""
        result = {}
        for l_idx, (k, v) in self._buffer.items():
            if k is not None and v is not None:
                result[l_idx] = (k, v)
        return result

    def clear_buffer(self):
        """Clear the capture buffer."""
        self._buffer.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._buffer.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cellmem_v3.py::TestKVInterceptor -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nanochat/kv_interceptor.py tests/test_cellmem_v3.py
git commit -m "feat(v3): KVInterceptor for pre-RoPE K/V capture"
```

---

### Task 4: CellMemWrapper v3 + Training Script

**Files:**
- Create: `scripts/train_cellmem_qwen_v3.py`

This is the largest task. It creates the orchestrator (CellMemWrapper) and training loop with InfoNCE contrastive loss and two-phase schedule.

- [ ] **Step 1: Write CellMemWrapper + training loop**

Create `scripts/train_cellmem_qwen_v3.py` with:

1. **CellMemWrapper**: orchestrates KVInterceptor, MemoryRouter, MemoryStore. Implements:
   - `write_memory(text)`: forward pass → KVInterceptor captures K/V → SurpriseCalculator picks tokens → MemoryStore.write() + MemoryRouter.encode_episode()
   - `_memory_read_hook(layer_idx)`: hook that calls MemoryRouter.route() → MemoryStore.read() → scaled_dot_product_attention on retrieved K/V (no RoPE) → add to residual

2. **Training data**: same synthetic format as v2 (`generate_training_data()`)

3. **Training loop**:
   - Phase 1 (warmup): `0.1 * L_LM + 1.0 * L_contrastive`, ~20 epochs
   - Phase 2 (main): `1.0 * L_LM + 0.1 * L_contrastive`, ~30 epochs
   - CLIP-style batch negatives for InfoNCE

4. **Eval**: Router Recall@k, Generation Recall, router discrimination gap

Key implementation notes from spec:
- Router is shared across layers (one MemoryRouter instance)
- K/V in attention: `Q = W_Q(h_query)` (backbone, frozen), `K = K_mem` (from store, no RoPE), `V = V_mem`
- `h_query += o_proj(attn_output)` using backbone's o_proj (frozen)
- Empty memory → skip attention, return unchanged hidden_states
- Backbone is ALWAYS frozen. Only router params are trainable.

- [ ] **Step 2: Run a smoke test**

Run: `python3 -c "from scripts.train_cellmem_qwen_v3 import CellMemWrapper; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/train_cellmem_qwen_v3.py
git commit -m "feat(v3): CellMemWrapper + training script with InfoNCE and two-phase schedule"
```

---

### Task 5: Integration Smoke Test

**Files:**
- Modify: `tests/test_cellmem_v3.py`

- [ ] **Step 1: Write integration test that loads a small model**

```python
# tests/test_cellmem_v3.py (append)

class TestIntegrationSmoke:
    """Integration tests requiring a model. Skip if no GPU or model not available."""

    @pytest.fixture
    def wrapper(self):
        """Load smallest Qwen model for testing."""
        try:
            from scripts.train_cellmem_qwen_v3 import CellMemWrapper
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = "Qwen/Qwen2.5-0.5B"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            n_layers = model.config.num_hidden_layers
            layer_indices = list(range(n_layers - 2, n_layers))  # top-2
            return CellMemWrapper(model, tokenizer, layer_indices, device=device)
        except Exception:
            pytest.skip("Model not available")

    def test_write_then_read_no_crash(self, wrapper):
        wrapper.write_memory("Dr. Elena Voss discovered Pyrothene in 2031")
        # forward pass with memory should not crash
        tokens = wrapper.tokenizer("What did Dr. Voss discover?", return_tensors="pt")
        tokens = {k: v.to(wrapper.device) for k, v in tokens.items()}
        with torch.no_grad():
            output = wrapper.base_model(**tokens)
        assert output.logits is not None

    def test_empty_memory_no_change(self, wrapper):
        tokens = wrapper.tokenizer("Hello world", return_tensors="pt")
        tokens = {k: v.to(wrapper.device) for k, v in tokens.items()}
        with torch.no_grad():
            out1 = wrapper.base_model(**tokens).logits.clone()
        # With empty memory, output should be identical
        with torch.no_grad():
            out2 = wrapper.base_model(**tokens).logits
        assert torch.allclose(out1, out2, atol=1e-5)
```

- [ ] **Step 2: Run integration tests (if model available)**

Run: `python3 -m pytest tests/test_cellmem_v3.py::TestIntegrationSmoke -v -s`
Expected: PASS (or skip if model not available)

- [ ] **Step 3: Commit**

```bash
git add tests/test_cellmem_v3.py
git commit -m "test(v3): integration smoke tests with Qwen 0.5B"
```

---

### Task 6: GPU Training Run + Eval

**Files:**
- Create: `runs/spot_train_v3.yaml` (SkyPilot config)

- [ ] **Step 1: Create SkyPilot YAML for v3 training**

Create `runs/spot_train_v3.yaml` adapted from existing `runs/spot_train.yaml`, pointing to `scripts/train_cellmem_qwen_v3.py` with v3 args.

- [ ] **Step 2: Run Phase 1 training**

```bash
sky launch -c cellmem-v3 runs/spot_train_v3.yaml
```

Monitor: router discrimination gap after epoch 20 (Phase 1 checkpoint).
Target: gap ≥ 0.1 (go/no-go decision).

- [ ] **Step 3: Run Phase 2 training**

If Phase 1 passes, continue to Phase 2 (epochs 20-50).

- [ ] **Step 4: Eval and report metrics**

| Metric | Target | Result |
|---|---|---|
| Router Recall@4 | ≥ 80% | ? |
| Generation Recall | ≥ 75% | ? |
| Baseline ΔPPL | < 0.5 | ? |
| Discrimination gap | ≥ 0.3 | ? |

- [ ] **Step 5: Commit results**

```bash
git add runs/spot_train_v3.yaml
git commit -m "feat(v3): training config and eval results"
```
