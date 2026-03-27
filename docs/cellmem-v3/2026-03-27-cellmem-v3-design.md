# CellMem v3 — Design Spec

> **Status:** Approved design, pending implementation.
> **Date:** 2026-03-27
> **Branch:** `cellmem/v2` → will become `cellmem/v3`
> **Predecessor:** CellMem v2 (ContentGate + LoRA + hidden state storage)

## Goal

Replace CellMem v2's failed ContentGate (scalar gate that couldn't discriminate positive/negative examples) with a router-based retrieval system that stores K/V pairs instead of hidden states. Preserve CellMem's unique properties: surprise-gated writes, cross-session persistence, inference-time encoding.

## Why v2 Failed

The ContentGate (tonic + phasic scalar gate) could not learn to discriminate between relevant and irrelevant memory, regardless of training strategy:

| Strategy tried | Result | Root cause |
|---|---|---|
| Cascade training | Gate saturated at 1.0 | Gate + LoRA compete for gradient |
| Annealing (gate frozen → unfrozen) | Gate stuck at 0.03 | Gradient diluted through 18 frozen layers |
| LoRA frozen + 200x gate LR | Gate collapsed to 0.1 | LM gradient says "close gate, model does better without memory" |
| BCE gate supervision + LoRA frozen | Gate stuck at 0.1 | MLP found local optimum: close everything, 10% residual suffices |

The fundamental issues: (1) scalar gate has one gradient signal, not enough to discriminate, (2) LM loss gradient traverses 18 frozen layers before reaching the gate, (3) stored hidden states are too abstract ("blurry photocopies").

## Architecture Overview

### Write Path (during inference)

1. **Input tokens** processed by backbone normally
2. **KVInterceptor** hooks capture K/V **pre-RoPE** at router layers (default: top-4)
3. **SurpriseCalculator** computes per-token cross-entropy surprise
4. Surprising tokens (CE > threshold) → K/V saved to **MemoryStore** per-layer
5. Every 8 tokens → **MemoryRouter**.encode_episode() produces a router key (mean-pooled)

### Read Path (during generation)

1. Hook on router layer intercepts hidden_states h_query
2. **MemoryRouter**.route(h_query, router_keys) → top-k episode indices via cosine similarity
3. **MemoryStore**.read(top_k_indices) → original K/V pairs of selected episodes
4. Scaled dot-product attention: Q=W_Q(h_query), K=K_mem, V=V_mem (**no RoPE**)
5. h_query += o_proj(attn_output) → residual stream updated
6. Repeat for each router layer (each uses its own K/V bank)

### Memory Store Structure

```
Per-layer K/V bank:
  keys:   [n_layers, n_tokens, n_kv_heads, d_head]  # e.g. [4, 256, 4, 128]
  values: [n_layers, n_tokens, n_kv_heads, d_head]

Router keys (for episode retrieval):
  router_keys: [n_episodes, d_router]  # e.g. [32, 128]
  (mean-pool every episode_size tokens)

Metadata:
  surprise: [n_tokens]
  age: [n_tokens]
  write_ptr, active_count
```

Storage: ~0.5 MB for 256 tokens × 4 layers (vs 1.3 MB in v2 for hidden states).

## Key Design Decisions

### 1. Memory Granularity: Hybrid

Per-token K/V writes (maximum fidelity), but the router selects **episodes** (groups of 8 tokens). This combines the precision of individual K/V pairs with efficient routing over compressed representations.

- 256 tokens in memory → 32 episodes
- Router compares query vs 32 episode keys (efficient)
- Top-k episodes selected → attention on ~k×8 original K/V pairs (precise)

### 2. Layer Placement: Configurable, Default Top-4

Router layers are configurable via `router_layers` parameter. Default: last 4 layers of the model (e.g., layers 32-35 for Qwen3.5-4B with 36 layers).

Rationale: with SFT on small synthetic data, fewer layers converge better. MSA uses top-half (18 layers) but has 159B tokens of pre-training. We start conservative and can scale up.

### 3. K/V Storage: Per-Layer Pre-RoPE

Each router layer stores its own K/V pairs, captured **before RoPE** is applied. This means:

- K/V are position-agnostic ("timeless memory")
- No LoRA needed — K/V are already in the backbone's native format
- Each layer has its own "perspective" on the stored tokens
- Attention works on content similarity only (no positional confusion)

### 4. RoPE Handling: Pre-RoPE (Position-Agnostic)

Memory K/V are intercepted before rotary position embeddings are applied. At retrieval time, no RoPE is applied to memory K/V. Attention operates purely on semantic content.

Rationale: episodic memory is inherently "timeless" — we care about WHAT was said, not WHERE in the original sequence. If we discover that intra-episode ordering matters, we can upgrade to episode-wise RoPE (MSA approach).

### 5. No LoRA

Eliminated. The backbone's Q projection already knows how to attend to K/V it produced. The only new trainable parameters are the router projectors W_Q^R and W_K^R.

### 6. Router: Shared Across Layers

One router (W_Q^R, W_K^R) shared across all router layers. The decision "which episode is relevant" is semantic and global — each layer then uses its own K/V from the selected episodes.

Total new parameters: ~656K (2 × Linear(2560, 128)), or 0.016% of the model.

## Components

### CellMemConfig (dataclass)

Extended from v2 with new fields:

```python
# New in v3
router_dim: int = 128           # dimension of router embedding space
router_layers: str = "top4"     # "top3" | "top4" | "top8" | "top_half"
episode_size: int = 8           # tokens per episode for router keys
top_k: int = 4                  # episodes to retrieve
contrastive_tau: float = 0.07   # InfoNCE temperature
```

### MemoryRouter (nn.Module) — NEW

```python
W_Q_R: Linear(d_model, d_router)  # query projector
W_K_R: Linear(d_model, d_router)  # key projector

route(h_query, router_keys) → top_k_indices, scores
encode_episode(h_tokens) → router_key [d_router]
contrastive_loss(q, positive, negatives) → InfoNCE scalar
```

Init: `normal_(std=0.02)` — never zero (v2 lesson: zero-init blocks gradient via chain rule).

### MemoryStore (class, stateful) — EVOLVED

Same structure as v2 (slots + surprise + age + write_ptr) but stores K/V per-layer instead of hidden states, plus router keys per episode.

```python
write(kv_per_layer, router_key, surprise)
read(episode_indices) → (keys, values) for selected episodes
get_router_keys() → [n_episodes, d_router]
save() / load() / snapshot()  # cross-session persistence
```

Keeps: decorrelation gate (min_novelty), delta writes, decay on load, surprise-based eviction.

### KVInterceptor (class, hook-based) — NEW

Captures K/V pre-RoPE during forward pass via PyTorch hooks. Model-specific (Qwen implementation first).

```python
register_hooks(model, layer_indices)
get_buffered_kv() → {layer_idx: (K, V)} for surprising tokens
clear_buffer()
remove_hooks()
```

Separated from CellMemWrapper for model-agnosticism — support other architectures by swapping only the interceptor.

### SurpriseCalculator — UNCHANGED from v2

### CellMemWrapper (nn.Module) — REWRITTEN

Orchestrator that connects all components. No LoRA, no ContentGate.

## Training

### Loss Function: InfoNCE Contrastive

```
L_aux = -log(exp(cos(Q^R, K^R_positive) / τ) / Σ_i exp(cos(Q^R, K^R_i) / τ))
```

- τ = 0.07 (temperature)
- In a batch of B examples, each query has 1 positive episode and B-1 natural negatives (CLIP-style, no explicit negative pairs needed)
- Gradients flow through d_router dimensions (rich signal, unlike v2's scalar BCE)

### Two-Phase Schedule (MSA-style)

**Phase 1 — Warmup (~20 epochs):**
- Loss: `0.1 * L_LM + 1.0 * L_contrastive`
- Trainable: router W_Q^R, W_K^R only
- Backbone: frozen
- Goal: router learns "this episode contains the answer to this query"

**Phase 2 — Main (~30 epochs):**
- Loss: `1.0 * L_LM + 0.1 * L_contrastive`
- Trainable: router W_Q^R, W_K^R (fine-tune)
- Backbone: frozen
- Goal: model generates correct answer using retrieved K/V

### Training Data

Same synthetic data as v2 (fictional facts + queries). Structure per example:

```json
{
  "memory": "Dr. Elena Voss discovered Pyrothene in 2031 at CERN",
  "query": "What did Dr. Voss discover?",
  "answer": "Pyrothene",
  "type": "positive"
}
```

Negatives come from batch (other examples' episodes), no explicit negative pairs needed.

## Techniques Inventory

### Kept from v2 (6)
- **Surprise-gated writes (BTSP)** — NIMH/Scripps 2025
- **SurpriseCalculator** — NIMH/Scripps 2025
- **Decay on load** — Original CellMem
- **Snapshot + rollback** — Original CellMem
- **Decorrelation gate** — Neuroscience (DG pattern separation)
- **Delta writes** — Infini-attention (Munkhdalai et al. 2024)

### Evolved (5)
- **MemoryStore** → K/V per-layer pre-RoPE (from MSA)
- **Two-phase training** → MSA-style warmup/main schedule
- **Auxiliary loss** → InfoNCE contrastive (from MSA + LeWorldModel principle)
- **Init ≠ zero** → applies to W_Q^R, W_K^R (v2 lesson)
- **Max vector norm** → to validate on K/V

### Dropped (5)
- **ContentGate** — replaced by router (gradient dilution, collapse)
- **Pupil range [0.1, 0.9]** — no sigmoid to protect
- **Canal lock clamp** — no scalar base to clamp
- **LoRA** — K/V pre-RoPE are already native format
- **MemoryRMSNorm** — K/V already normalized by backbone layer norm

## Testing

### Unit Tests (CPU, no model)

**MemoryRouter** (~8 tests): route() correctness, encode_episode() shape, contrastive_loss() gradient flow, init non-zero.

**MemoryStore v3** (~9 tests): write K/V per-layer, read by episode indices, router_keys shape, episodic grouping, full-overwrite eviction, decorrelation, save/load roundtrip, decay.

**KVInterceptor** (~4 tests): capture shape, pre-RoPE verification, buffer clear, hook removal.

### Integration Tests (GPU)

- Write-then-read roundtrip
- Cross-session persistence
- Router discriminates correct episode (>50% accuracy)
- Forward pass with non-empty memory doesn't crash
- No baseline degradation (memory empty → identical output)

### Success Metrics

| Metric | Target | v2 baseline |
|---|---|---|
| **Router Recall@4** (primary) | ≥ 80% | N/A (no router) |
| **Generation Recall** | ≥ 75% | 63.6% |
| **Baseline Preservation** (ΔPPL) | < 0.5 | — |
| **Router Discrimination** (pos-neg gap) | ≥ 0.3 | ~0 (gate couldn't discriminate) |

**First checkpoint:** router discrimination. If the router doesn't discriminate pos/neg (gap < 0.1), stop and investigate. Unlike v2's scalar gate, InfoNCE has rich gradient in embedding space — this should work.

## File Map

| File | Contents | Notes |
|---|---|---|
| `nanochat/cellmem_v3.py` | CellMemConfig, MemoryStore, SurpriseCalculator, MemoryRouter | Core module, no HF dependency |
| `nanochat/kv_interceptor.py` | KVInterceptor | Hook-based K/V capture, Qwen-specific |
| `scripts/train_cellmem_qwen.py` | CellMemWrapper, training loop, eval | Rewritten for v3 |
| `tests/test_cellmem_v3.py` | Unit tests for all components | New file |

## References

- **MSA** — Memory Sparse Attention (arXiv:2603.23516): Router architecture, K/V storage, InfoNCE, two-phase training
- **LeWorldModel** (arXiv:2603.19312): SIGReg principle — direct auxiliary loss for desired properties
- **Video Reasoning** (arXiv:2603.16870): Reasoning localized in middle-to-upper layers
- **NIMH/Scripps 2025** (Bhatt et al.): BTSP anti-Hebbian plasticity, surprise-gated writes
- **Infini-attention** (Munkhdalai et al. 2024, arXiv:2404.07143): Delta update rule
- **LoRA** (Hu et al. 2021): Low-rank adaptation (used in v2, dropped in v3)

## Visual Reference

See `docs/cellmem-v3/techniques-inventory.html` for interactive technique inventory with sources.
