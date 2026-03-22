# nanochat/cellmem_v2.py
"""
CellMem v2: Persistent inference-time memory for nanochat.

Memory vectors live in the transformer's d_model space, accumulate during
inference via anti-Hebbian surprise-gated writes, and persist across invocations.

Spec: docs/superpowers/specs/2026-03-22-cellmem-v2-design.md
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import torch


@dataclass
class CellMemConfig:
    enabled: bool = False
    n_slots: int = 64
    write_strategy: str = "per_token"    # "per_token" | "chunk"
    chunk_size: int = 64
    surprise_threshold: float = 4.0
    layers: str = "last3"                # "last3" | "mid" | "all"
    decay_factor: float = 0.95           # applied once per load (per session)
    max_snapshots: int = 5
    memory_dir: str = "~/.cache/nanochat/memory"
    max_vector_norm: float = 50.0


class MemoryStore:
    """Stateful container for K memory vectors in R^d_model."""

    def __init__(self, config: CellMemConfig, d_model: int):
        self.config = config
        self.d_model = d_model
        K = config.n_slots
        self.vectors = torch.zeros(K, d_model)
        self.surprise = torch.zeros(K)
        self.age = torch.zeros(K)
        self.write_ptr = 0
        self.active_count = 0
        self.last_written_pos = -1  # for token deduplication in generation

    @torch.compiler.disable
    def write(self, vector: torch.Tensor, surprise: float):
        """Write a memory vector to the next available slot.
        When full, overwrite the slot with lowest surprise score."""
        K = self.config.n_slots
        # Clamp vector norm to prevent explosion
        vec_norm = vector.norm()
        if vec_norm > self.config.max_vector_norm:
            vector = vector * (self.config.max_vector_norm / vec_norm)

        if self.active_count < K:
            # Still have empty slots
            idx = self.write_ptr
            self.write_ptr = (self.write_ptr + 1) % K
            self.active_count += 1
        else:
            # Full: overwrite slot with lowest surprise
            idx = self.surprise[:K].argmin().item()

        self.vectors[idx] = vector.detach()
        self.surprise[idx] = surprise
        self.age[idx] = 0

    @torch.compiler.disable
    def read(self):
        """Return (vectors, mask) or None if empty.
        mask is a bool tensor indicating active slots."""
        if self.active_count == 0:
            return None
        mask = torch.zeros(self.config.n_slots, dtype=torch.bool)
        if self.active_count < self.config.n_slots:
            mask[:self.active_count] = True
        else:
            mask[:] = True
        return self.vectors, mask

    def age_all(self, tokens_processed: int = 1):
        """Increment age of all active memories."""
        if self.active_count > 0:
            self.age[:self.active_count] += tokens_processed
