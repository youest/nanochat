# CellMem v2 — Design Specification

## Goal

Add a persistent, inference-time memory system to nanochat's GPT model. Memory vectors live in the transformer's embedding space (d_model=768), accumulate during inference via anti-Hebbian surprise-gated writes, and persist across invocations. Inspired by NIMH/Scripps 2025 neuroscience findings on prediction-error-driven synaptic plasticity.

## Architecture Overview

CellMem v2 is a key-value memory store of 64 vectors in R^768 that integrates into the transformer via a **separate cross-attention pass** (tokens attend to memories). A per-layer learnable gate (init=0) controls how much the model reads from memory. Memories are written when prediction error (cross-entropy loss) exceeds a threshold — the anti-Hebbian principle: remember what you can't predict.

## Principles (from NIMH/Scripps 2025 paper)

1. **Memory IS the structural change** — not a separate database. Memory vectors live in the same space as transformer hidden states.
2. **Anti-Hebbian** — write on prediction error, not on repetition. High surprise = write.
3. **During experience** — memory forms during inference, not training.
4. **Decay** — memories fade over time, simulating biological forgetting.

## Components

### 1. MemoryStore (`nanochat/cellmem_v2.py`)

Stateful container for K=64 memory vectors.

**State:**
- `vectors: Tensor[K, d_model]` — the memory vectors (768-dim each)
- `surprise: Tensor[K]` — surprise score when each memory was written
- `age: Tensor[K]` — age in tokens since write
- `write_ptr: int` — next slot to write (circular)
- `active_count: int` — number of slots actually written (0 to K)

**Write policy:**
When surprise exceeds threshold θ, write the hidden state into the next available slot. When full (active_count == K), overwrite the oldest slot with lowest surprise (least informative memory).

**Read interface:**
Returns `(vectors, mask)` where mask indicates which slots are active (have been written to). When `active_count == 0` (empty memory), returns `None` — caller skips memory integration entirely (equivalent to baseline model).

**Persistence:**
- `save(path)` — save vectors + metadata to .pt file
- `load(path)` — load and apply decay: `vectors *= decay_factor` (one decay step per load, i.e. per inference session)
- `snapshot(path)` — save timestamped backup for rollback
- Max 5 snapshots retained (oldest auto-deleted)

**All mutating methods** (`write`, `load`, `save`, `snapshot`) must be decorated with `@torch.compiler.disable` since they mutate state incompatible with torch.compile tracing.

### 2. Attention Integration (modify `CausalSelfAttention` in `nanochat/gpt.py`)

**Problem:** Flash Attention's `causal=True` mode requires Q and K to have the same sequence length. Concatenating memory K/V with token K/V breaks causal masking and sliding window computation.

**Solution:** Two-pass attention. The normal causal self-attention is unchanged. A separate cross-attention pass reads from memory with no causal mask and no sliding window. The two outputs are combined with a learnable gate.

**Detailed pipeline for layers with memory:**
```
# === Pass 1: Normal causal self-attention (UNCHANGED) ===
Q = c_q(x)                                    # [B, T, H, D]
K_tok = c_k(x)                                # [B, T, Hkv, D]
V_tok = c_v(x)                                # [B, T, Hkv, D]
# Value residual (ve) applied to V_tok only, not V_mem
if ve is not None:
    V_tok = V_tok + gate * ve
Q = apply_rotary_emb(Q, cos, sin)
K_tok = apply_rotary_emb(K_tok, cos, sin)
Q, K_tok = norm(Q), norm(K_tok)
Q, K_tok = Q * 1.2, K_tok * 1.2
attn_tok = flash_attn(Q, K_tok, V_tok, causal=True, window_size=window_size)

# === Pass 2: Cross-attention to memory ===
# Reuse same W_k, W_v but NO RoPE (memories have no position)
K_mem = c_k(mem_vectors)                       # [1, K_active, Hkv, D]
V_mem = c_v(mem_vectors)                       # [1, K_active, Hkv, D]
K_mem = norm(K_mem)                            # QK-norm (same as tokens)
K_mem = K_mem * 1.2                            # same scaling as tokens
K_mem = K_mem.expand(B, -1, -1, -1)           # broadcast to batch
V_mem = V_mem.expand(B, -1, -1, -1)
# Use F.scaled_dot_product_attention directly (not flash_attn wrapper)
# because the SDPA fallback in flash_attention.py hard-codes causal=True.
# This is simple non-causal cross-attention over K=64 vectors — no kernel needed.
attn_mem = F.scaled_dot_product_attention(Q, K_mem, V_mem, is_causal=False)  # non-causal

# === Combine ===
gate = sigmoid(mem_gate[layer_idx])            # mem_gate init to -10, sigmoid(-10)≈0
output = attn_tok + gate * attn_mem
output = c_proj(output)                        # project back to residual stream
```

**Key details:**
- RoPE is NOT applied to K_mem (memories have no sequential position)
- QK-norm and 1.2 scaling ARE applied to K_mem (same space as token keys)
- Value embeddings (ve) are NOT added to V_mem (memories are already learned representations)
- Sliding window does NOT affect memory attention (separate pass, `causal=False`)
- `mem_gate` is initialized to -10.0 so `sigmoid(-10) ≈ 0.0000454` — effectively zero at start

**Cost:** For K=64 and T=2048, the memory cross-attention is ~3.1% of the token self-attention per layer. For "last3" config (3 layers), total overhead is <1%.

### 3. Surprise Calculator (`nanochat/cellmem_v2.py`)

Computes prediction error for the write decision. Used only during inference.

**Per-token strategy:**
For each token position i, compute `surprise_i = -log P(t_i | t_{<i})` (cross-entropy loss). If `surprise_i > θ`, write `hidden_state[i]` to memory.

**Chunk-based strategy:**
Process tokens in chunks of `chunk_size` (default 64). At end of each chunk, compute mean surprise over the chunk. If above θ, write the mean hidden state of the chunk to memory.

Both strategies implemented, selectable via config. Comparative eval determines which is better.

**Token deduplication:** Track `last_written_pos` to avoid writing the same token's memory multiple times during naive generation (which recomputes the full sequence each step). Only tokens at positions > `last_written_pos` are candidates for writing.

### 4. CellMemConfig

```python
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
```

### 5. GPT Integration Points (modify `nanochat/gpt.py`)

**GPTConfig:** Add `cellmem: CellMemConfig` field (default: disabled).

**GPT.__init__:** If cellmem enabled, create:
- `self.memory_store = MemoryStore(config.cellmem)` — NOT a nn.Module, just a stateful container
- `self.mem_gates = nn.ParameterList([nn.Parameter(torch.tensor([-10.0])) for _ in selected_layers])`
- Number of gate params = number of selected layers (3 for "last3", 1 for "mid", 12 for "all")

**GPT.init_weights:** mem_gates initialized to -10.0 (sigmoid ≈ 0).

**GPT.forward:**
- **Training mode (self.training=True):** Skip memory integration entirely. mem_gates exist as parameters but receive no gradients (no memory path is executed). The model trains identically to baseline. Gates will be fine-tuned in a separate post-training phase with synthetic memory data.
- **Inference mode:** If memory_store has active memories (active_count > 0):
  - Pre-project memory vectors through each selected layer's c_k/c_v
  - Pass K_mem/V_mem + gate to each selected layer's attention
  - After forward: compute per-token surprise from logits vs actual tokens
  - Conditionally write new memories to memory_store

**GPT.setup_optimizer:** Add mem_gates to AdamW scalar group (not Muon). During base training these params exist but are inert (no grad flow).

**GPT.generate:**
- At start: load memory from disk if available (applies decay)
- Track token positions to avoid duplicate memory writes across generation steps
- At end: save memory + snapshot to disk

### 6. Persistence Manager (`nanochat/cellmem_v2.py`)

Handles save/load/snapshot lifecycle.

**Save format (.pt):**
```python
{
    "vectors": Tensor[K, d_model],
    "surprise": Tensor[K],
    "age": Tensor[K],
    "write_ptr": int,
    "active_count": int,
    "config": dict,
    "version": 2,
    "saved_at": ISO8601 timestamp,
}
```

**Snapshot naming:** `snapshot_{ISO8601_timestamp}.pt`

**Decay on load:** `vectors *= decay_factor` — one decay step per session (one call to load = one session). Simple and predictable.

## Eval Plan

Three decisions require empirical comparison:

### Eval 1: Write Strategy
- **A**: per-token surprise-gated
- **C**: chunk-based surprise-gated (chunk_size=64)
- **Metric**: cross-sequence fact recall accuracy (novel facts in session 1, query in session 2)
- **Baseline**: no memory (cellmem disabled)

### Eval 2: Layer Placement
- **last3**: layers 9, 10, 11
- **mid**: layer 6
- **all**: layers 0-11
- **Metric**: same as Eval 1 + compute cost measurement

### Eval 3: Best Combination
- 2 write strategies x 3 layer placements = 6 configurations
- Run each on same eval set, pick winner by accuracy/cost tradeoff

### Eval 4 (follow-up, if results promising): Memory Capacity
- K=64 vs K=128 vs K=256 with winning write strategy + layer config
- Only run if Eval 1-3 show positive signal over baseline

### Eval Protocol
1. Train base model normally (no cellmem — training is identical to baseline)
2. Optional: fine-tune mem_gates with synthetic memory data
3. Run inference eval script:
   - Phase 1 (LEARN): feed novel facts, memories accumulate
   - Phase 2 (INTERFERE): feed unrelated text, test memory robustness
   - Phase 3 (RECALL): query facts from Phase 1, measure accuracy
4. Compare 6 configurations + baseline (no memory)

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `nanochat/cellmem_v2.py` | Create | MemoryStore, SurpriseCalculator, PersistenceManager |
| `nanochat/gpt.py` | Modify | GPTConfig, attention integration, gate params, optimizer |
| `tests/test_cellmem_v2.py` | Create | Unit tests for MemoryStore, write/read/persist |
| `tests/test_gpt_cellmem_v2.py` | Create | Integration tests for GPT + CellMem |
| `scripts/eval_cellmem_v2.py` | Create | Eval script for 6-config comparison |
| `scripts/base_train.py` | Modify | Add --cellmem CLI flags |

## Non-Goals

- **Training-time memory writes**: CellMem v2 writes only during inference. During training the model trains identically to baseline (gates are inert). Gate fine-tuning is a separate optional phase.
- **MemoryBank / RAG**: No separate retrieval system. The memory IS the 64 vectors.
- **Matrix M from v1**: Replaced entirely by vector-based approach.

## Constraints / Lessons from v1

- `torch.compile` is incompatible with stateful ops — `MemoryStore.write()`, `MemoryStore.read()`, `MemoryStore.load()`, `MemoryStore.save()`, and the surprise gating logic must use `@torch.compiler.disable`
- CellMem params (mem_gates) must use AdamW, not Muon (Muon requires stackable same-shape tensors)
- Gate params should init to effectively zero (sigmoid(-10) ≈ 0, model starts identical to baseline)
- Memory vectors need norm control to prevent explosion (clamp vector norms after write)
