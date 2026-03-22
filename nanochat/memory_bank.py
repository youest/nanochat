"""
MemoryBank: persistent embedding store for cross-invocation memory.
Stores (key, value) pairs where key is in the transformer's embedding space
and value is CellMem's output. Retrieval via dot-product attention on a
FAISS-filtered working set.
"""

from dataclasses import dataclass
import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


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

    def decay(self):
        """Age all slots and decay usage. Call once per invocation."""
        if self.size == 0:
            return
        self._age += 1
        self._usage *= self.config.decay_usage

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
            indices = torch.from_numpy(np.asarray(indices)).long()
        else:
            # Brute-force for small banks
            sims = torch.mv(self._keys, query) / (self._keys.norm(dim=1) * query.norm() + 1e-8)
            indices = sims.topk(k).indices

        return self._keys[indices], self._values[indices]

    def read_from_working_set(self, query: torch.Tensor, ws_keys: torch.Tensor, ws_values: torch.Tensor) -> torch.Tensor:
        """Read from a pre-selected working set. Use for per-token reads."""
        if ws_keys.shape[0] == 0:
            return torch.zeros(self.config.d_value)
        query = query.detach()
        d = self.config.d_key ** 0.5
        scores = ws_keys @ query / d
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(1) * ws_values).sum(dim=0)
