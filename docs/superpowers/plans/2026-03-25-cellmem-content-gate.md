# CellMem v2: Content-Dependent Gate + Mixed Training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CellMem work in interactive chat by replacing the blind scalar gate with a content-dependent gate, training with negative examples, and adding ACh-inspired read/write coupling.

**Architecture:** The scalar gate `sigmoid(s)` is replaced by a content-dependent `sigmoid(MLP([h_local; h_mem; h_local - h_mem]))` that learns WHEN memory is useful. Training includes positive examples (memory needed), negative examples (memory irrelevant), and poisoned examples (memory wrong). ACh coupling suppresses reads during high-surprise writes.

**Tech Stack:** PyTorch, HuggingFace transformers (Qwen3), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `nanochat/cellmem_v2.py` | Modify | Add `ContentGate` class, `MemoryRMSNorm` |
| `scripts/train_cellmem_qwen.py` | Modify | Use ContentGate in CellMemWrapper, mixed training, ACh coupling, conversational data |
| `tests/test_cellmem_v2.py` | Modify | Tests for ContentGate, MemoryRMSNorm |
| `tests/test_cellmem_wrapper.py` | Create | Tests for CellMemWrapper integration, data gen, training loop, eval |

---

### Task 1: ContentGate module

The core fix. Replaces a fixed scalar with an MLP that compares local hidden state vs memory retrieval output. This is the CA1 comparator from the literature review (section 8.4).

**Files:**
- Modify: `nanochat/cellmem_v2.py` (add ContentGate class after CellMemConfig)
- Modify: `tests/test_cellmem_v2.py` (add TestContentGate class)

- [ ] **Step 1: Write failing tests for ContentGate**

```python
# tests/test_cellmem_v2.py — add import at top of file (after existing imports):
import torch.nn.functional as F

# Then add at bottom of file:

class TestContentGate:
    """Content-dependent gate (CA1 comparator): gate = sigmoid(MLP([h_local; h_mem; h_local - h_mem]))"""

    def test_output_shape(self):
        """Gate produces per-token scalar in [0, 1]."""
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=64)
        h_local = torch.randn(2, 10, 64)   # [B, T, D]
        h_mem = torch.randn(2, 10, 64)     # [B, T, D]
        g = gate(h_local, h_mem)
        assert g.shape == (2, 10, 1), f"Expected (2, 10, 1), got {g.shape}"
        assert (g >= 0).all() and (g <= 1).all(), "Gate values must be in [0, 1]"

    def test_gradient_flows(self):
        """MLP parameters receive gradients."""
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=32)
        h_local = torch.randn(1, 4, 32, requires_grad=True)
        h_mem = torch.randn(1, 4, 32, requires_grad=True)
        g = gate(h_local, h_mem)
        g.sum().backward()
        for name, p in gate.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradient"

    def test_irrelevant_memory_gates_low(self):
        """When h_mem is random noise unrelated to h_local, gate should be ~0.5 at init
        (MLP output near 0 -> sigmoid ~ 0.5). After training on negatives it would go lower."""
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=64)
        h_local = torch.randn(1, 8, 64)
        h_mem = torch.randn(1, 8, 64)  # unrelated noise
        g = gate(h_local, h_mem)
        # At init, MLP output ~ 0, so sigmoid ~ 0.5. Check it's reasonable.
        assert g.mean().item() == pytest.approx(0.5, abs=0.15)

    def test_init_near_zero_output(self):
        """MLP last layer initialized to zeros -> output starts at sigmoid(0) = 0.5.
        This ensures the model starts identical to having no memory (neutral gate)."""
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=128)
        h_local = torch.randn(1, 1, 128)
        h_mem = torch.randn(1, 1, 128)
        g = gate(h_local, h_mem)
        assert g.item() == pytest.approx(0.5, abs=0.01), \
            f"Gate should start at ~0.5 (zero-init last layer), got {g.item()}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_v2.py::TestContentGate -v`
Expected: FAIL with "cannot import name 'ContentGate'"

- [ ] **Step 3: Implement ContentGate**

Add to `nanochat/cellmem_v2.py` after `CellMemConfig`:

```python
class ContentGate(torch.nn.Module):
    """Content-dependent gate (CA1 comparator).

    Compares local hidden state with memory retrieval output to decide
    whether memory is useful for the current token.

    gate = sigmoid(W2 @ GELU(W1 @ [h_local; h_mem; h_local - h_mem]))

    Initialized so output starts at sigmoid(0) = 0.5 (neutral).
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Input: [h_local; h_mem; h_local - h_mem] -> 3 * d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 * d_model, d_model // 4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model // 4, 1),
        )
        # Zero-init last layer -> sigmoid(0) = 0.5 at start
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_local: torch.Tensor, h_mem: torch.Tensor) -> torch.Tensor:
        """Returns gate values in [0, 1], shape [B, T, 1]."""
        x = torch.cat([h_local, h_mem, h_local - h_mem], dim=-1)
        return torch.sigmoid(self.net(x))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cellmem_v2.py::TestContentGate -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add nanochat/cellmem_v2.py tests/test_cellmem_v2.py
git commit -m "feat: ContentGate module (CA1 comparator) with tests"
```

---

### Task 2: MemoryRMSNorm

Normalize memory vectors before cross-attention read, preventing high-magnitude vectors from dominating. From AttnRes paper (Kimi/Moonshot, section 4 of literature review).

**Files:**
- Modify: `nanochat/cellmem_v2.py` (add MemoryRMSNorm)
- Modify: `tests/test_cellmem_v2.py` (add TestMemoryRMSNorm)

- [ ] **Step 1: Write failing tests for MemoryRMSNorm**

```python
# tests/test_cellmem_v2.py — add at bottom (F already imported in Task 1)

class TestMemoryRMSNorm:
    """RMSNorm on memory vectors before cross-attention read."""

    def test_output_shape_preserved(self):
        from nanochat.cellmem_v2 import MemoryRMSNorm
        norm = MemoryRMSNorm(d_model=64)
        x = torch.randn(8, 64)  # [N_slots, D]
        y = norm(x)
        assert y.shape == x.shape

    def test_normalizes_magnitude(self):
        """Vectors with different magnitudes should have similar RMS after normalization."""
        from nanochat.cellmem_v2 import MemoryRMSNorm
        norm = MemoryRMSNorm(d_model=64)
        x = torch.randn(4, 64)
        x[0] *= 100  # one vector much larger
        y = norm(x)
        rms = (y ** 2).mean(dim=-1).sqrt()
        # All RMS values should be similar (within 50% of each other)
        assert rms.max() / rms.min() < 1.5

    def test_preserves_direction(self):
        """Normalization should preserve cosine similarity structure."""
        from nanochat.cellmem_v2 import MemoryRMSNorm
        norm = MemoryRMSNorm(d_model=32)
        x = torch.randn(3, 32)
        y = norm(x)
        cos_before = F.cosine_similarity(x[0:1], x[1:2])
        cos_after = F.cosine_similarity(y[0:1], y[1:2])
        assert cos_before.item() == pytest.approx(cos_after.item(), abs=0.01)

    def test_gradient_flows(self):
        from nanochat.cellmem_v2 import MemoryRMSNorm
        norm = MemoryRMSNorm(d_model=16)
        x = torch.randn(4, 16, requires_grad=True)
        y = norm(x)
        y.sum().backward()
        assert x.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_v2.py::TestMemoryRMSNorm -v`
Expected: FAIL with "cannot import name 'MemoryRMSNorm'"

- [ ] **Step 3: Implement MemoryRMSNorm**

Add to `nanochat/cellmem_v2.py` after `ContentGate`:

```python
class MemoryRMSNorm(torch.nn.Module):
    """RMSNorm for memory vectors before cross-attention read.

    Prevents high-magnitude vectors from dominating retrieval.
    From AttnRes (Kimi/Moonshot, 2026): normalize values before attention.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        return (x / rms) * self.weight
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cellmem_v2.py::TestMemoryRMSNorm -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add nanochat/cellmem_v2.py tests/test_cellmem_v2.py
git commit -m "feat: MemoryRMSNorm for magnitude-invariant retrieval"
```

---

### Task 3: Integrate ContentGate + RMSNorm into CellMemWrapper

Replace the scalar gate with ContentGate and add RMSNorm before memory projection in `CellMemWrapper`. This is the critical integration task.

**Files:**
- Modify: `scripts/train_cellmem_qwen.py` (CellMemWrapper class, entire file)
- Create: `tests/test_cellmem_wrapper.py`

- [ ] **Step 1: Write failing tests that verify CellMemWrapper uses ContentGate**

These tests must FAIL before the implementation (unlike Task 1-2 which test standalone modules).

```python
# tests/test_cellmem_wrapper.py
"""Tests for CellMemWrapper components (no GPU required)."""
import torch
import torch.nn.functional as F
import pytest


class TestCellMemWrapperUsesContentGate:
    """Verify CellMemWrapper uses ContentGate instead of scalar gate."""

    def test_content_gate_produces_per_token_values(self):
        """ContentGate should produce [B, T, 1] gate values, not a fixed scalar."""
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=64)
        h_local = torch.randn(2, 8, 64)
        h_mem = torch.randn(2, 8, 64)
        g = gate(h_local, h_mem)
        # Per-token gate: shape [B, T, 1]
        assert g.shape == (2, 8, 1)
        # Different tokens can have different gate values
        assert not torch.allclose(g[0, 0], g[0, -1]), \
            "Different tokens should get different gate values"

    def test_content_gate_trainable_param_count(self):
        """ContentGate should have a reasonable number of trainable params."""
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=256)
        n_params = sum(p.numel() for p in gate.parameters())
        # 3*256*64 + 64 + 64*1 + 1 = ~49K params
        assert n_params > 1000     # non-trivial
        assert n_params < 200_000  # not too large

    def test_rms_norm_equalizes_attention(self):
        """RMSNorm should make attention weights more uniform across memory slots."""
        from nanochat.cellmem_v2 import MemoryRMSNorm
        d = 64
        norm = MemoryRMSNorm(d_model=d)
        mem = torch.randn(5, d)
        mem[0] *= 100  # one vector much larger
        query = torch.randn(1, d)

        # Without norm: biased toward large vector
        scores_raw = (query @ mem.T) / (d ** 0.5)
        attn_raw = torch.softmax(scores_raw, dim=-1)
        entropy_raw = -(attn_raw * attn_raw.log()).sum()

        # With norm: more uniform
        mem_normed = norm(mem)
        scores_normed = (query @ mem_normed.T) / (d ** 0.5)
        attn_normed = torch.softmax(scores_normed, dim=-1)
        entropy_normed = -(attn_normed * attn_normed.log()).sum()

        assert entropy_normed > entropy_raw


class TestMemoryRMSNormIntegration:
    """Verify RMSNorm equalizes memory vector magnitudes for fairer attention."""

    def test_equalized_attention_weights(self):
        """With RMSNorm, attention entropy should be higher (more uniform)."""
        from nanochat.cellmem_v2 import MemoryRMSNorm
        d = 64
        norm = MemoryRMSNorm(d_model=d)

        # Memory: one vector 100x larger than others
        mem = torch.randn(5, d)
        mem[0] *= 100

        query = torch.randn(1, d)

        # Attention without norm (biased toward large vector)
        scores_raw = (query @ mem.T) / (d ** 0.5)
        attn_raw = torch.softmax(scores_raw, dim=-1)
        entropy_raw = -(attn_raw * attn_raw.log()).sum()

        # Attention with norm (more uniform)
        mem_normed = norm(mem)
        scores_normed = (query @ mem_normed.T) / (d ** 0.5)
        attn_normed = torch.softmax(scores_normed, dim=-1)
        entropy_normed = -(attn_normed * attn_normed.log()).sum()

        assert entropy_normed > entropy_raw, \
            f"RMSNorm should increase attention entropy: {entropy_normed:.3f} vs {entropy_raw:.3f}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestCellMemWrapperUsesContentGate -v`
Expected: FAIL — source still contains `mem_gates`, not `content_gates`

- [ ] **Step 3: Replace scalar gate with ContentGate + RMSNorm in CellMemWrapper**

In `scripts/train_cellmem_qwen.py`, make these changes:

**3a. Add import at top of file (after existing imports):**
```python
from nanochat.cellmem_v2 import ContentGate, MemoryRMSNorm
```

**3b. Replace `__init__` gate setup (lines 71-81). Remove:**
```python
        self.mem_gates = nn.ParameterList()
```
**Replace with:**
```python
        self.content_gates = nn.ModuleList()
        self.mem_rms_norm = MemoryRMSNorm(self.hidden_size).to(device=device, dtype=torch.bfloat16)
```

**Replace the for-loop body (lines 74-81). Remove:**
```python
        for idx in layer_indices:
            # Gate: init to 0 -> sigmoid(0) = 0.5
            self.mem_gates.append(nn.Parameter(torch.zeros(1, device=device)))

            # LoRA on q_proj and v_proj of this layer's attention
            attn = self._get_attn_module(idx)
            self.lora_layers[f"{idx}_q"] = LoRALinear(attn.q_proj, rank=lora_rank).to(device=device, dtype=torch.bfloat16)
            self.lora_layers[f"{idx}_v"] = LoRALinear(attn.v_proj, rank=lora_rank).to(device=device, dtype=torch.bfloat16)
```
**Replace with:**
```python
        for idx in layer_indices:
            # ContentGate instead of scalar
            self.content_gates.append(
                ContentGate(self.hidden_size).to(device=device, dtype=torch.bfloat16)
            )
            # LoRA on q_proj and v_proj of this layer's attention
            attn = self._get_attn_module(idx)
            self.lora_layers[f"{idx}_q"] = LoRALinear(attn.q_proj, rank=lora_rank).to(device=device, dtype=torch.bfloat16)
            self.lora_layers[f"{idx}_v"] = LoRALinear(attn.v_proj, rank=lora_rank).to(device=device, dtype=torch.bfloat16)
```

**3c. Replace `_cross_attend_to_memory` method entirely (lines 131-165):**
```python
    def _cross_attend_to_memory(self, hidden_states, gate_idx, layer_idx):
        """Compute gated cross-attention from hidden_states to memory vectors.
        Uses ContentGate (CA1 comparator) and MemoryRMSNorm."""
        B, T, C = hidden_states.shape
        attn = self._get_attn_module(layer_idx)

        # Q from hidden states (through LoRA-adapted projection)
        q_proj = self.lora_layers[f"{layer_idx}_q"]
        Q = q_proj(hidden_states)  # [B, T, num_heads * head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # RMSNorm on memory before projection (prevents magnitude bias)
        mem = self.mem_rms_norm(self.memory_vectors[:self.memory_count])
        mem = mem.unsqueeze(0).expand(B, -1, -1)
        mem = mem.to(hidden_states.dtype).to(hidden_states.device)

        K_mem = attn.k_proj(mem)
        V_mem_proj = self.lora_layers[f"{layer_idx}_v"]
        V_mem = V_mem_proj(mem)

        N_mem = self.memory_count
        K_mem = K_mem.view(B, N_mem, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_mem = V_mem.view(B, N_mem, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Cross-attention (no causal mask — memory has no position)
        enable_gqa = self.num_heads != self.num_kv_heads
        y_mem = F.scaled_dot_product_attention(
            Q, K_mem, V_mem, is_causal=False, enable_gqa=enable_gqa
        )
        attn_out_dim = self.num_heads * self.head_dim
        y_mem = y_mem.transpose(1, 2).contiguous().view(B, T, attn_out_dim)
        y_mem = attn.o_proj(y_mem)

        # Content-dependent gate (replaces scalar gate)
        gate = self.content_gates[gate_idx](hidden_states, y_mem)  # [B, T, 1]
        self._last_gate_values.append(gate)  # collect for gate supervision loss

        return gate * y_mem
```

**3d. Add gate value collection for supervision. After `self.memory_count = 0` (line 85-86), add:**
```python
        self._last_gate_values = []  # collected during forward for gate supervision loss
```

**3e. Replace `trainable_parameters` method (lines 248-252):**
```python
    def trainable_parameters(self):
        """Return only trainable parameters (content gates + RMSNorm + LoRA)."""
        params = []
        for cg in self.content_gates:
            params.extend(cg.parameters())
        params.extend(self.mem_rms_norm.parameters())
        params.extend(self.lora_layers.parameters())
        return params
```

**3f. Replace `save_lora` method (lines 258-267):**
```python
    def save_lora(self, path):
        """Save LoRA weights, content gates, and RMSNorm."""
        state = {
            "content_gates": [cg.state_dict() for cg in self.content_gates],
            "mem_rms_norm": self.mem_rms_norm.state_dict(),
            "lora_layers": self.lora_layers.state_dict(),
            "layer_indices": self.layer_indices,
            "lora_rank": self.lora_layers[list(self.lora_layers.keys())[0]].lora_A.out_features,
            "version": 2,  # v2 = content gate format
        }
        torch.save(state, path)
        print(f"Saved LoRA weights to {path}")
```

**3g. Replace `load_lora` method (lines 269-275):**
```python
    def load_lora(self, path):
        """Load LoRA weights, content gates, and RMSNorm."""
        state = torch.load(path, weights_only=False, map_location=self.device)
        if state.get("version", 1) >= 2:
            # v2 format: content gates
            for i, cg in enumerate(self.content_gates):
                cg.load_state_dict(state["content_gates"][i])
            self.mem_rms_norm.load_state_dict(state["mem_rms_norm"])
        else:
            # v1 format: scalar gates — ignore, ContentGate starts at 0.5 which is fine
            print("Warning: loading v1 checkpoint (scalar gates) into v2 (content gates). Gates reset to neutral.")
        self.lora_layers.load_state_dict(state["lora_layers"])
        print(f"Loaded LoRA weights from {path}")
```

**3h. Fix remaining `wrapper.mem_gates` references outside the training loop.**

Note: The `gate_vals` line inside the training loop (line 600) and summary (line 672) will be replaced entirely when Task 5 rewrites the training loop. Only fix the lines that Task 5 does NOT touch.

Line 745 (serve_web_ui): Replace:
```python
print(f"Model ready. Gates: {[f'{torch.sigmoid(g).item():.3f}' for g in wrapper.mem_gates]}")
```
With:
```python
print(f"Model ready. Content gates: {len(wrapper.content_gates)} layers")
```

Line 910 (health endpoint): Replace:
```python
"gate": f"{torch.sigmoid(wrapper.mem_gates[0]).item():.3f}"
```
With:
```python
"gate_type": "content_dependent", "n_gate_layers": len(wrapper.content_gates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cellmem_wrapper.py tests/test_cellmem_v2.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_cellmem_qwen.py tests/test_cellmem_wrapper.py
git commit -m "feat: integrate ContentGate + RMSNorm + ACh coupling into CellMemWrapper"
```

---

### Task 4: Negative + poisoned training data

The model needs to see examples where memory is NOT useful (negatives) and examples where memory is WRONG (poisoned). Without these, the gate never learns to close.

**Files:**
- Modify: `scripts/train_cellmem_qwen.py` (add `_generate_mixed_data` after `_split_data`)
- Modify: `tests/test_cellmem_wrapper.py` (add `TestMixedTrainingData`)

- [ ] **Step 1: Write failing tests for data generation**

```python
# tests/test_cellmem_wrapper.py — add at bottom

class TestMixedTrainingData:
    """Mixed training data: positive, negative, poisoned examples."""

    def test_negative_data_has_irrelevant_memory(self):
        """Negative examples: query + memory that's unrelated."""
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=50, seed=42)
        negatives = [d for d in data if d["type"] == "negative"]
        assert len(negatives) > 0
        for neg in negatives:
            assert "context" in neg       # irrelevant context (goes to memory)
            assert "query" in neg         # query
            assert "answer" in neg        # correct answer (model should answer WITHOUT memory)
            assert neg["type"] == "negative"

    def test_poisoned_data_has_wrong_memory(self):
        """Poisoned examples: memory contains wrong answer."""
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=50, seed=42)
        poisoned = [d for d in data if d["type"] == "poisoned"]
        assert len(poisoned) > 0
        for p in poisoned:
            assert "context" in p         # wrong context
            assert "query" in p
            assert "answer" in p          # correct answer
            assert "wrong_context" in p   # the misleading context

    def test_data_ratios(self):
        """Data should be ~40% positive, 40% negative, 20% poisoned."""
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=100, seed=42)
        counts = {"positive": 0, "negative": 0, "poisoned": 0}
        for d in data:
            counts[d["type"]] += 1
        assert counts["positive"] >= 30   # ~40%
        assert counts["negative"] >= 30   # ~40%
        assert counts["poisoned"] >= 10   # ~20%

    def test_positive_data_matches_original_format(self):
        """Positive examples should have same format as _generate_procedural_data."""
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=20, seed=42)
        positives = [d for d in data if d["type"] == "positive"]
        assert len(positives) > 0
        for p in positives:
            assert "context" in p
            assert "query" in p
            assert "answer" in p
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestMixedTrainingData -v`
Expected: FAIL with "cannot import name '_generate_mixed_data'"

- [ ] **Step 3: Implement _generate_mixed_data**

Add to `scripts/train_cellmem_qwen.py` after `_split_data` function (line 455):

```python
def _generate_mixed_data(n=200, seed=42):
    """Generate mixed training data: positive, negative, poisoned.

    - Positive (40%): context matches query. Memory IS useful.
    - Negative (40%): context is IRRELEVANT to query. Memory should be ignored.
    - Poisoned (20%): context contains WRONG answer. Model must not trust memory blindly.
    """
    rng = _random.Random(seed)
    all_procedural = _generate_procedural_data(n * 2, seed=seed)

    # Split into source pools
    pool_a = all_procedural[:n]
    pool_b = all_procedural[n:]

    data = []
    for i in range(n):
        roll = rng.random()
        if roll < 0.4:
            # Positive: matching context + query
            ex = pool_a[i % len(pool_a)]
            data.append({**ex, "type": "positive"})
        elif roll < 0.8:
            # Negative: query from pool_a, context from pool_b (unrelated)
            ex_q = pool_a[i % len(pool_a)]
            ex_c = pool_b[rng.randint(0, len(pool_b) - 1)]
            data.append({
                "context": ex_c["context"],  # irrelevant context
                "query": ex_q["query"],
                "answer": ex_q["answer"],    # correct answer
                "type": "negative",
            })
        else:
            # Poisoned: query from pool_a, context has wrong answer
            ex_q = pool_a[i % len(pool_a)]
            ex_wrong = pool_b[rng.randint(0, len(pool_b) - 1)]
            data.append({
                "context": ex_wrong["context"],  # misleading context
                "wrong_context": ex_wrong["context"],
                "query": ex_q["query"],
                "answer": ex_q["answer"],  # correct answer (not from wrong context)
                "type": "poisoned",
            })
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestMixedTrainingData -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_cellmem_qwen.py tests/test_cellmem_wrapper.py
git commit -m "feat: mixed training data (positive/negative/poisoned)"
```

---

### Task 5: Mixed training loop with gate supervision loss

Modify the training loop to handle three types of examples. Use **gate supervision loss** instead of KL divergence — directly penalize gate values for negatives (gate should be 0) and reward them for positives (gate should be 1). This is a denser, more direct signal inspired by MOPD (Nemotron-Cascade 2).

**Files:**
- Modify: `scripts/train_cellmem_qwen.py` (add `compute_gate_loss`, modify `run_experiment`)
- Modify: `tests/test_cellmem_wrapper.py`

- [ ] **Step 1: Write failing tests for gate supervision loss**

```python
# tests/test_cellmem_wrapper.py — add at bottom

class TestGateSupervisionLoss:
    """Gate supervision: directly penalize gate values for negatives/positives."""

    def test_compute_gate_loss_negatives(self):
        """For negatives, gate should be pushed to 0."""
        from scripts.train_cellmem_qwen import compute_gate_loss
        gate_values = torch.tensor([[[0.8]], [[0.6]], [[0.9]]])  # [B, T, 1]
        loss = compute_gate_loss(gate_values, target="close")
        assert loss.shape == ()
        assert loss.item() > 0.5  # gates are high, loss should be high

    def test_compute_gate_loss_positives(self):
        """For positives, gate should be pushed to 1."""
        from scripts.train_cellmem_qwen import compute_gate_loss
        gate_values = torch.tensor([[[0.2]], [[0.3]], [[0.1]]])  # [B, T, 1]
        loss = compute_gate_loss(gate_values, target="open")
        assert loss.shape == ()
        assert loss.item() > 0.5  # gates are low, loss should be high

    def test_gate_loss_zero_when_correct(self):
        """Loss should be near 0 when gate matches target."""
        from scripts.train_cellmem_qwen import compute_gate_loss
        # Gate close to 0 + target close -> low loss
        gate_close = torch.tensor([[[0.01]], [[0.02]]])
        loss_close = compute_gate_loss(gate_close, target="close")
        assert loss_close.item() < 0.05

        # Gate close to 1 + target open -> low loss
        gate_open = torch.tensor([[[0.98]], [[0.99]]])
        loss_open = compute_gate_loss(gate_open, target="open")
        assert loss_open.item() < 0.05

    def test_gate_loss_gradient_flows(self):
        """Gate loss must backprop to gate parameters."""
        from scripts.train_cellmem_qwen import compute_gate_loss
        gate_values = torch.tensor([[[0.7]]], requires_grad=True)
        loss = compute_gate_loss(gate_values, target="close")
        loss.backward()
        assert gate_values.grad is not None
        assert gate_values.grad.abs().sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestGateSupervisionLoss -v`
Expected: FAIL with "cannot import name 'compute_gate_loss'"

- [ ] **Step 3: Implement compute_gate_loss**

Add to `scripts/train_cellmem_qwen.py` after `compute_retrieval_loss`:

```python
def compute_gate_loss(gate_values, target="close"):
    """Gate supervision loss: directly push gate values toward target.

    For negatives: target="close" -> loss = gate.mean() (penalize gate > 0)
    For positives: target="open"  -> loss = (1 - gate).mean() (penalize gate < 1)

    This is denser and more direct than KL divergence on logits.
    Inspired by MOPD (Nemotron-Cascade 2) dense supervision principle.
    """
    if target == "close":
        return gate_values.mean()
    elif target == "open":
        return (1.0 - gate_values).mean()
    else:
        raise ValueError(f"target must be 'close' or 'open', got {target}")
```

- [ ] **Step 4: Replace data generation and training loop in run_experiment**

**4a. Replace the module-level data globals (lines 458-460).** Delete:
```python
_ALL_DATA = _generate_procedural_data(200, seed=42)
TRAIN_DATA, TEST_DATA = _split_data(_ALL_DATA, test_ratio=0.2, seed=42)
```

**4b. In `run_experiment`, before the training loop (after `print(f"Trainable parameters: ...")` around line 525), add data generation:**
```python
    # Generate mixed training data (positive + negative + poisoned)
    n_examples = getattr(args, 'n_examples', 200)
    mixed_data = _generate_mixed_data(n_examples, seed=42)
    train_data, test_data = _split_data(mixed_data, test_ratio=0.2, seed=42)
    print(f"\nData: {len(train_data)} train, {len(test_data)} test")
```

**4c. Add gate value collection to CellMemWrapper.** The ContentGate produces [B, T, 1] values during forward. We need to capture them for the gate supervision loss. Add to `CellMemWrapper.__init__`:
```python
        self._last_gate_values = []  # collected during forward for gate supervision
```

In `_cross_attend_to_memory`, after computing `gate = self.content_gates[gate_idx](hidden_states, y_mem)`, add:
```python
        self._last_gate_values.append(gate)
```

**4d. Replace the entire old training loop** (from `for epoch in range(args.epochs):` through the `gate_vals` print block):

Replace with:
```python
    for epoch in range(args.epochs):
        total_loss = 0.0
        n = 0
        epoch_data = list(train_data)
        _random.shuffle(epoch_data)

        for ex in epoch_data:
            wrapper.clear_memory()
            wrapper._last_gate_values = []  # reset gate collection
            wrapper.write_memory_selective(ex["context"], top_k=args.top_k)

            if wrapper.memory_count == 0:
                continue

            ex_type = ex.get("type", "positive")
            optimizer.zero_grad()

            if ex_type == "positive":
                # Positive: LM loss (memory should help) + gate open supervision
                lm_loss = compute_retrieval_loss(wrapper, tokenizer,
                                                  ex["query"], ex["answer"], device)
                gate_vals = wrapper._last_gate_values
                g_loss = compute_gate_loss(torch.cat(gate_vals, dim=1), target="open") if gate_vals else 0.0
                loss = lm_loss + 0.1 * g_loss

            elif ex_type == "negative":
                # Negative: only gate supervision (gate should close)
                prompt = ex["query"] + " " + ex["answer"]
                tokens = tokenizer(prompt, return_tensors="pt").to(device)
                wrapper(input_ids=tokens["input_ids"])  # forward to collect gate values
                gate_vals = wrapper._last_gate_values
                loss = compute_gate_loss(torch.cat(gate_vals, dim=1), target="close") if gate_vals else torch.tensor(0.0)

            elif ex_type == "poisoned":
                # Poisoned: LM loss on correct answer + gate close (don't trust memory)
                lm_loss = compute_retrieval_loss(wrapper, tokenizer,
                                                  ex["query"], ex["answer"], device)
                gate_vals = wrapper._last_gate_values
                g_loss = compute_gate_loss(torch.cat(gate_vals, dim=1), target="close") if gate_vals else 0.0
                loss = lm_loss + 0.1 * g_loss
            else:
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | n={n}")
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_cellmem_wrapper.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/train_cellmem_qwen.py tests/test_cellmem_wrapper.py
git commit -m "feat: gate supervision loss for mixed training (positive/negative/poisoned)"
```

---

### ~~Task 6: ACh read/write coupling~~ — SKIPPED

**Reason:** The surprise_level is set during `write_memory_selective` and reset to 0.0 before the forward pass. The coupling has NO EFFECT in the current architecture (write and read happen in separate phases, not in parallel). Deferred to a future iteration when the architecture supports concurrent read/write.

---

### ~~Task 7: Conversational training data~~ — SKIPPED

**Reason:** `_generate_conversations` would be dead code — the training loop in Task 5 uses `_generate_mixed_data`, and nothing integrates conversations. Will revisit after evaluating mixed training results.

---

### Task 8: Comprehensive eval framework

Combine all metrics: recall, divergence, poisoned resistance, multi-memory selectivity.

**Files:**
- Modify: `scripts/train_cellmem_qwen.py` (add `eval_comprehensive`)
- Modify: `tests/test_cellmem_wrapper.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cellmem_wrapper.py — add at bottom

class TestEvalMetrics:
    """Comprehensive eval returns structured metrics."""

    def test_eval_comprehensive_signature(self):
        """eval_comprehensive should have the expected signature and return keys."""
        from scripts.train_cellmem_qwen import eval_comprehensive
        import inspect
        sig = inspect.signature(eval_comprehensive)
        params = list(sig.parameters.keys())
        assert "wrapper" in params
        assert "tokenizer" in params
        assert "data" in params
        assert "device" in params

    def test_eval_comprehensive_docstring_mentions_all_metrics(self):
        """Docstring should document all 5 return metrics."""
        from scripts.train_cellmem_qwen import eval_comprehensive
        doc = eval_comprehensive.__doc__
        assert doc is not None
        for key in ["positive_recall", "poisoned_resistance",
                     "multi_memory_recall", "mean_gate_positive", "mean_gate_negative"]:
            assert key in doc, f"Docstring missing metric: {key}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestEvalMetrics -v`
Expected: FAIL with "cannot import name 'eval_comprehensive'"

- [ ] **Step 3: Implement eval_comprehensive**

Add to `scripts/train_cellmem_qwen.py` after `compute_gate_loss`:

```python
def eval_comprehensive(wrapper, tokenizer, data, device, show=3):
    """Comprehensive evaluation across positive, negative, and poisoned examples.

    Returns dict with keys:
        positive_recall: % of positive examples where answer keywords found
        poisoned_resistance: % of poisoned examples where CORRECT answer given
        multi_memory_recall: % recall with 5 accumulated memories
        mean_gate_positive: avg gate value on positives (should be high ~1.0)
        mean_gate_negative: avg gate value on negatives (should be low ~0.0)
    """
    stopwords = {"the", "a", "an", "is", "was", "are", "of", "in", "to", "and",
                 "that", "it", "for", "on", "with"}
    results = {"positive": [], "negative": [], "poisoned": []}
    gate_values = {"positive": [], "negative": [], "poisoned": []}

    for ex in data:
        ex_type = ex.get("type", "positive")
        if ex_type not in results:
            continue
        wrapper.clear_memory()
        wrapper._last_gate_values = []
        wrapper.write_memory_selective(ex["context"], top_k=8)
        if wrapper.memory_count == 0:
            continue

        generated = wrapper.generate(ex["query"], max_new_tokens=32)

        # Collect gate values
        if wrapper._last_gate_values:
            mean_g = torch.cat(wrapper._last_gate_values, dim=1).mean().item()
            gate_values[ex_type].append(mean_g)

        answer_words = set(ex["answer"].lower().split()) - stopwords
        gen_lower = generated.lower()
        matched = sum(1 for w in answer_words if w in gen_lower)
        score = matched / max(len(answer_words), 1)
        results[ex_type].append(score > 0.5)

    pos_recall = sum(results["positive"]) / max(len(results["positive"]), 1) * 100
    poison_resist = sum(results["poisoned"]) / max(len(results["poisoned"]), 1) * 100

    mean_gate_pos = sum(gate_values["positive"]) / max(len(gate_values["positive"]), 1)
    mean_gate_neg = sum(gate_values["negative"]) / max(len(gate_values["negative"]), 1)

    # Multi-memory recall
    positives = [e for e in data if e.get("type") == "positive"]
    multi_hits = 0
    multi_total = 0
    for g_start in range(0, min(len(positives), 20), 5):
        group = positives[g_start:g_start + 5]
        wrapper.clear_memory()
        for ex in group:
            wrapper.write_memory_selective(ex["context"], top_k=8)
        for ex in group:
            generated = wrapper.generate(ex["query"], max_new_tokens=32)
            answer_words = set(ex["answer"].lower().split()) - stopwords
            gen_lower = generated.lower()
            matched = sum(1 for w in answer_words if w in gen_lower)
            if matched / max(len(answer_words), 1) > 0.5:
                multi_hits += 1
            multi_total += 1
    multi_recall = multi_hits / max(multi_total, 1) * 100

    metrics = {
        "positive_recall": pos_recall,
        "poisoned_resistance": poison_resist,
        "multi_memory_recall": multi_recall,
        "mean_gate_positive": mean_gate_pos,
        "mean_gate_negative": mean_gate_neg,
    }

    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVAL")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if "gate" in k:
            print(f"  {k:25s} {v:.4f}")
        else:
            print(f"  {k:25s} {v:.1f}%")

    return metrics
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestEvalMetrics -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_cellmem_qwen.py tests/test_cellmem_wrapper.py
git commit -m "feat: comprehensive eval (recall + divergence + poison + multi-mem + F1)"
```

---

### Task 9: Wire up run_experiment with mixed training + comprehensive eval

Update the main flow to use mixed data and comprehensive eval. Also add `--n-examples` CLI arg.

**Files:**
- Modify: `scripts/train_cellmem_qwen.py` (run_experiment, argparse)
- Modify: `tests/test_cellmem_wrapper.py`

- [ ] **Step 1: Write failing test for --n-examples arg**

```python
# tests/test_cellmem_wrapper.py — add at bottom

class TestRunExperimentWiring:
    """Verify run_experiment uses mixed data and comprehensive eval."""

    def test_n_examples_arg_exists(self):
        """CLI should accept --n-examples argument."""
        from scripts.train_cellmem_qwen import main
        import inspect
        source = inspect.getsource(main)
        assert 'n-examples' in source or 'n_examples' in source, \
            "main() must add --n-examples argument to argparse"

    def test_run_experiment_uses_mixed_data(self):
        """run_experiment should call _generate_mixed_data, not just _generate_procedural_data."""
        import inspect
        from scripts.train_cellmem_qwen import run_experiment
        source = inspect.getsource(run_experiment)
        assert '_generate_mixed_data' in source, \
            "run_experiment must use _generate_mixed_data for training data"

    def test_run_experiment_calls_eval_comprehensive(self):
        """run_experiment should use eval_comprehensive for evaluation."""
        import inspect
        from scripts.train_cellmem_qwen import run_experiment
        source = inspect.getsource(run_experiment)
        assert 'eval_comprehensive' in source, \
            "run_experiment must call eval_comprehensive"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestRunExperimentWiring -v`
Expected: FAIL — source doesn't contain these references yet

- [ ] **Step 3: Modify run_experiment and main**

Note: Task 5 already replaced the module-level data globals with local `train_data`/`test_data` and rewrote the training loop. This task only needs to:
1. Replace remaining `TRAIN_DATA`/`TEST_DATA` references (in eval phases and baseline)
2. Replace the old eval phases with `eval_comprehensive`
3. Add `--n-examples` to argparse
4. Update the data generation to use `args.n_examples`

**3a. In `run_experiment`, update the data generation (added in Task 5 Step 4b) to use args:**
Replace `n_examples = getattr(args, 'n_examples', 200)` with `n_examples = args.n_examples`.

**3b. Replace remaining TRAIN_DATA/TEST_DATA references** in `run_experiment` (in baseline, eval, ablation, multi-memory phases — everything outside the training loop). Search for `TRAIN_DATA` and `TEST_DATA` and replace with `train_data` and `test_data`.

**3c. Replace the eval phases (PHASE 3 through PHASE 5, and the summary block)** with a single call. Find the code block starting with `print("PHASE 3: EVAL")` through the end of `print(f"Ablation (no mem): {ablation_pct:.1f}%")` and replace with:
```python
    # --- EVAL ---
    print("\n" + "=" * 60)
    print("PHASE 3: COMPREHENSIVE EVAL")
    print("=" * 60)
    metrics = eval_comprehensive(wrapper, tokenizer, test_data, device)
```

**3e. Add `--n-examples` to argparse in `main()` (line 687-704).** Add after `--lora-rank`:
```python
    parser.add_argument("--n-examples", type=int, default=300,
                        help="Number of mixed training examples to generate")
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_cellmem_wrapper.py tests/test_cellmem_v2.py -v --ignore=tests/test_gpt.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_cellmem_qwen.py tests/test_cellmem_wrapper.py
git commit -m "feat: wire up mixed training + comprehensive eval in run_experiment"
```

---

### Task 10: Update SkyPilot YAML and deploy

**Files:**
- Modify: `runs/spot_eval_cellmem.yaml`

- [ ] **Step 1: Update YAML run command**

```yaml
run: |
  cd ~/nanochat
  source .venv/bin/activate
  git fetch && git checkout cellmem/v2 && git pull

  export PATH=$HOME/.local/bin:$PATH
  uv pip install "transformers @ git+https://github.com/huggingface/transformers.git" accelerate 2>&1 | tail -3

  echo "=== CellMem v2: Content-Dependent Gate Training ==="
  PYTHONUNBUFFERED=1 python -m scripts.train_cellmem_qwen \
    --model Qwen/Qwen3-4B \
    --epochs 30 \
    --lr 5e-4 \
    --layers mid \
    --n-examples 300 \
    --save-path ~/cellmem_lora_v2.pt \
    --serve \
    --port 8001
```

- [ ] **Step 2: Commit and push**

```bash
git add runs/spot_eval_cellmem.yaml
git commit -m "chore: update SkyPilot YAML for content-dependent gate training"
git push origin cellmem/v2
```

---

## Success Criteria

After all tasks, running `eval_comprehensive` should show:

| Metric | Baseline (scalar gate) | Target (content gate) |
|--------|----------------------|----------------------|
| Positive recall | 65% | >= 60% |
| Mean gate (positives) | ~0.5 (fixed) | > 0.7 (opens for memory) |
| Mean gate (negatives) | ~0.5 (fixed) | < 0.2 (closes for noise) |
| Poisoned resistance | ~0% (trusts blindly) | >= 50% |
| Multi-memory recall | ~40% | >= 50% |
| Chat quality | degenerates | coherent |

The key metric is **mean gate on negatives**: if this drops from ~0.5 to < 0.2, it means the gate has learned to close when memory is irrelevant — the root cause of chat degradation. Combined with mean gate on positives > 0.7, we'd know the gate is discriminating, not just always-closed.
