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
        self.episode_texts: list[str | None] = [None] * E
        self.surprise = torch.zeros(N)
        self.age = torch.zeros(N)
        self.write_ptr = 0
        self.active_count = 0
        self.episode_ptr = 0
        self.active_episodes = 0

    @torch.compiler.disable
    def write(self, kv: dict, router_key: torch.Tensor | None,
              surprise: float, text: str | None = None):
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
            self.episode_texts[ep_idx] = text

    def read_texts(self, episode_indices: list[int]) -> list[str]:
        """Return non-None texts for the given episode indices."""
        return [self.episode_texts[i] for i in episode_indices
                if i < len(self.episode_texts) and self.episode_texts[i] is not None]

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
            "episode_texts": self.episode_texts,
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
        E = self.config.n_slots // self.config.episode_size
        self.episode_texts = data.get("episode_texts", [None] * E)
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
