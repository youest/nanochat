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
import torch.nn.functional as F


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
    write_mode: str = "raw"              # "raw" | "delta" (prediction-error writes)
    delta_threshold: float = 0.1         # min delta norm to write (skip if below)
    min_novelty: float = 0.0             # min cosine distance to existing memories (0=off, 0.1=skip if >0.9 sim)


class ContentGate(torch.nn.Module):
    """Content-dependent gate (CA1 comparator).
    gate = sigmoid(W2 @ GELU(W1 @ [h_local; h_mem; h_local - h_mem]))
    Initialized so output starts at sigmoid(0) = 0.5 (neutral).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 * d_model, d_model // 4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model // 4, 1),
        )
        torch.nn.init.normal_(self.net[-1].weight, std=1e-3)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_local: torch.Tensor, h_mem: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_local, h_mem, h_local - h_mem], dim=-1)
        return torch.sigmoid(self.net(x))


class MemoryRMSNorm(torch.nn.Module):
    """RMSNorm for memory vectors before cross-attention read."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        return (x / rms) * self.weight


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

    def _memory_prediction(self, query: torch.Tensor) -> torch.Tensor:
        """What does memory already predict for this query?
        Uses cosine-similarity-weighted sum of active memories.
        Returns zero vector if memory is empty."""
        if self.active_count == 0:
            return torch.zeros_like(query)
        active = self.vectors[:self.active_count].to(query.device)  # [A, d]
        sim = F.cosine_similarity(query.unsqueeze(0), active, dim=-1)  # [A]
        weights = F.softmax(sim, dim=0)                     # [A]
        return (weights.unsqueeze(-1) * active).sum(0)      # [d]

    @torch.compiler.disable
    def write(self, vector: torch.Tensor, surprise: float):
        """Write a memory vector to the next available slot.
        When full, overwrite the slot with lowest surprise score.
        In delta mode, writes prediction error instead of raw vector."""
        K = self.config.n_slots

        # Decorrelation gate (DG pattern separation): skip if too similar to existing memories
        if self.config.min_novelty > 0 and self.active_count > 0:
            active = self.vectors[:self.active_count].to(vector.device)
            sims = F.cosine_similarity(vector.unsqueeze(0), active, dim=-1)  # [A]
            if sims.max().item() > (1.0 - self.config.min_novelty):
                return  # too similar, skip write

        # Delta update rule: write only the prediction error
        if self.config.write_mode == "delta":
            prediction = self._memory_prediction(vector)
            vector = vector - prediction
            # Skip if delta is below threshold (memory already knows this)
            if vector.norm().item() < self.config.delta_threshold:
                return

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

    @torch.compiler.disable
    def save(self, path):
        """Save memory state to .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vectors": self.vectors.clone(),
            "surprise": self.surprise.clone(),
            "age": self.age.clone(),
            "write_ptr": self.write_ptr,
            "active_count": self.active_count,
            "config": asdict(self.config),
            "version": 2,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(data, path)

    @torch.compiler.disable
    def load(self, path):
        """Load memory state from .pt file and apply decay."""
        path = Path(path)
        data = torch.load(path, weights_only=False)
        self.vectors = data["vectors"]
        self.surprise = data["surprise"]
        self.age = data["age"]
        self.write_ptr = data["write_ptr"]
        self.active_count = data["active_count"]
        # Apply decay
        if self.active_count > 0:
            self.vectors *= self.config.decay_factor

    @torch.compiler.disable
    def snapshot(self):
        """Save timestamped backup. Keep only max_snapshots most recent."""
        mem_dir = Path(self.config.memory_dir).expanduser()
        mem_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        snap_path = mem_dir / f"snapshot_{ts}.pt"
        self.save(snap_path)
        # Cleanup old snapshots
        snaps = sorted(mem_dir.glob("snapshot_*.pt"))
        while len(snaps) > self.config.max_snapshots:
            snaps.pop(0).unlink()


class SurpriseCalculator:
    """Computes per-token prediction error for memory write decisions."""

    def __init__(self, config: CellMemConfig):
        self.config = config

    def compute_surprise(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-token surprise (cross-entropy loss).
        Args:
            logits: [B, T, V] model predictions
            targets: [B, T] actual token ids
        Returns:
            surprises: [B, T] per-token loss values
        """
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            targets.view(B * T),
            reduction='none'
        )
        return loss.view(B, T)

    def get_write_mask(self, surprises: torch.Tensor, last_written_pos: int = -1) -> torch.Tensor:
        """Return boolean mask of positions exceeding surprise threshold.
        Positions <= last_written_pos are excluded (deduplication)."""
        mask = surprises > self.config.surprise_threshold
        if last_written_pos >= 0:
            B, T = mask.shape
            pos_mask = torch.arange(T, device=mask.device) > last_written_pos
            mask = mask & pos_mask.unsqueeze(0)
        return mask

    def get_chunk_surprises(self, surprises: torch.Tensor):
        """Split surprises into chunks, return list of chunk info dicts."""
        B, T = surprises.shape
        chunk_size = self.config.chunk_size
        chunks = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_surprise = surprises[:, start:end].mean().item()
            chunks.append({
                "start": start,
                "end": end,
                "mean_surprise": chunk_surprise,
            })
        return chunks
