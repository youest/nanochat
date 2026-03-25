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

    def test_wrapper_has_content_gates_attribute(self):
        """After integration, CellMemWrapper must have content_gates, not mem_gates."""
        from scripts.train_cellmem_qwen import CellMemWrapper
        assert hasattr(CellMemWrapper, '__init__')
        # We check the __init__ source for content_gates
        import inspect
        source = inspect.getsource(CellMemWrapper.__init__)
        assert 'content_gates' in source, \
            "CellMemWrapper.__init__ must create self.content_gates"
        assert 'mem_gates' not in source, \
            "CellMemWrapper.__init__ must NOT create self.mem_gates (replaced by content_gates)"

    def test_wrapper_has_mem_rms_norm(self):
        """CellMemWrapper must have mem_rms_norm attribute."""
        import inspect
        from scripts.train_cellmem_qwen import CellMemWrapper
        source = inspect.getsource(CellMemWrapper.__init__)
        assert 'mem_rms_norm' in source, \
            "CellMemWrapper.__init__ must create self.mem_rms_norm"

    def test_save_lora_includes_content_gates(self):
        """save_lora must serialize content_gates and mem_rms_norm."""
        import inspect
        from scripts.train_cellmem_qwen import CellMemWrapper
        source = inspect.getsource(CellMemWrapper.save_lora)
        assert 'content_gates' in source, "save_lora must include content_gates"
        assert 'mem_rms_norm' in source, "save_lora must include mem_rms_norm"


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

        # ACh coupling: suppress reads during high-surprise (encoding) moments
        ach_suppression = 1.0 - self.surprise_level
        gate = gate * ach_suppression

        return gate * y_mem
```

**3d. Add `surprise_level` init. After `self.memory_count = 0` (line 85-86), add:**
```python
        self.surprise_level = 0.0  # set during write, used by ACh coupling
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

### Task 5: Mixed training loop with divergence loss

Modify the training loop to handle three types of examples with appropriate losses.

**Files:**
- Modify: `scripts/train_cellmem_qwen.py` (add `compute_divergence_loss`, modify `run_experiment`)
- Modify: `tests/test_cellmem_wrapper.py`

- [ ] **Step 1: Write failing tests for divergence loss**

```python
# tests/test_cellmem_wrapper.py — add at bottom

class TestMixedTrainingLoss:
    """Training loss for different example types."""

    def test_compute_divergence_loss_shape(self):
        """Divergence loss returns a scalar."""
        from scripts.train_cellmem_qwen import compute_divergence_loss
        logits_with_mem = torch.randn(1, 10, 100)
        logits_without_mem = torch.randn(1, 10, 100)
        loss = compute_divergence_loss(logits_with_mem, logits_without_mem)
        assert loss.shape == ()  # scalar
        assert loss.item() >= 0  # KL divergence is non-negative

    def test_identical_logits_zero_divergence(self):
        """When logits are identical, divergence should be ~0."""
        from scripts.train_cellmem_qwen import compute_divergence_loss
        logits = torch.randn(1, 10, 100)
        loss = compute_divergence_loss(logits, logits.clone())
        assert loss.item() < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestMixedTrainingLoss -v`
Expected: FAIL with "cannot import name 'compute_divergence_loss'"

- [ ] **Step 3: Implement compute_divergence_loss**

Add to `scripts/train_cellmem_qwen.py` after `compute_retrieval_loss` (line 487):

```python
def compute_divergence_loss(logits_with_mem, logits_without_mem):
    """KL divergence between model output with vs without memory.
    For negative examples: this should be ~0 (memory shouldn't change output)."""
    p = F.log_softmax(logits_with_mem, dim=-1)
    q = F.softmax(logits_without_mem.detach(), dim=-1)
    return F.kl_div(p, q, reduction="batchmean")
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

**4c. Replace the entire old training loop** (from `for epoch in range(args.epochs):` through the `gate_vals` print block):

Old code to remove:
```python
    for epoch in range(args.epochs):
        total_loss = 0.0
        n = 0
        epoch_data = list(TRAIN_DATA)
        _random.shuffle(epoch_data)

        for ex in epoch_data:
            wrapper.clear_memory()
            wrapper.write_memory_selective(ex["context"], top_k=args.top_k)

            if wrapper.memory_count == 0:
                continue

            optimizer.zero_grad()
            loss = compute_retrieval_loss(wrapper, tokenizer,
                                          ex["query"], ex["answer"], device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        gate_vals = [f"{torch.sigmoid(g).item():.3f}" for g in wrapper.mem_gates]

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | gates={gate_vals} | n={n}")
```

Replace with:
```python
    for epoch in range(args.epochs):
        total_loss = 0.0
        n = 0
        epoch_data = list(train_data)
        _random.shuffle(epoch_data)

        for ex in epoch_data:
            wrapper.clear_memory()
            wrapper.write_memory_selective(ex["context"], top_k=args.top_k)

            if wrapper.memory_count == 0:
                continue

            ex_type = ex.get("type", "positive")

            if ex_type == "positive" or ex_type == "poisoned":
                # Positive: memory should help. Poisoned: should give correct answer anyway.
                optimizer.zero_grad()
                loss = compute_retrieval_loss(wrapper, tokenizer,
                                              ex["query"], ex["answer"], device)
            elif ex_type == "negative":
                # Negative: memory should NOT change output
                prompt = ex["query"] + " " + ex["answer"]
                tokens = tokenizer(prompt, return_tensors="pt").to(device)

                # Forward WITH memory
                out_with_mem = wrapper(input_ids=tokens["input_ids"])

                # Forward WITHOUT memory (temporarily disable)
                saved_count = wrapper.memory_count
                wrapper.memory_count = 0
                with torch.no_grad():
                    out_without_mem = wrapper(input_ids=tokens["input_ids"])
                wrapper.memory_count = saved_count

                optimizer.zero_grad()
                loss = compute_divergence_loss(out_with_mem.logits, out_without_mem.logits)
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
git commit -m "feat: mixed training loop with divergence loss for negatives"
```

---

### Task 6: ACh read/write coupling — surprise modulation during writes

When surprise is high (novel input, should be memorized), suppress memory reads. The ACh coupling is already wired into `_cross_attend_to_memory` (Task 3). Here we add the surprise level computation in `write_memory_selective`.

**Files:**
- Modify: `scripts/train_cellmem_qwen.py:186-219` (`write_memory_selective`)
- Modify: `tests/test_cellmem_wrapper.py`

- [ ] **Step 1: Write failing tests for ACh surprise setting**

```python
# tests/test_cellmem_wrapper.py — add at bottom

class TestAChCoupling:
    """ACh-inspired read/write coupling: high surprise suppresses reads."""

    def test_write_memory_selective_sets_surprise_level(self):
        """After write_memory_selective, surprise_level should be set > 0."""
        import inspect
        from scripts.train_cellmem_qwen import CellMemWrapper
        source = inspect.getsource(CellMemWrapper.write_memory_selective)
        assert 'surprise_level' in source, \
            "write_memory_selective must set self.surprise_level for ACh coupling"

    def test_surprise_level_reset_after_write(self):
        """surprise_level should be reset to 0 after write completes."""
        import inspect
        from scripts.train_cellmem_qwen import CellMemWrapper
        source = inspect.getsource(CellMemWrapper.write_memory_selective)
        # Check that surprise_level is reset at the end
        lines = source.split('\n')
        # Find last non-empty line that sets surprise_level
        set_lines = [i for i, l in enumerate(lines) if 'surprise_level' in l]
        assert len(set_lines) >= 2, \
            "write_memory_selective must set AND reset surprise_level"

    def test_ach_math_high_surprise_suppresses(self):
        """Pure math: with surprise_level=1.0, effective gate should be 0."""
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=32)
        h_local = torch.randn(1, 4, 32)
        h_mem = torch.randn(1, 4, 32)
        raw_gate = gate(h_local, h_mem)
        surprise_level = 1.0
        effective = raw_gate * (1.0 - surprise_level)
        assert effective.abs().max() < 0.01

    def test_ach_math_low_surprise_preserves(self):
        """Pure math: with surprise_level=0.0, effective gate equals raw gate."""
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=32)
        h_local = torch.randn(1, 4, 32)
        h_mem = torch.randn(1, 4, 32)
        raw_gate = gate(h_local, h_mem)
        surprise_level = 0.0
        effective = raw_gate * (1.0 - surprise_level)
        assert torch.allclose(effective, raw_gate)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestAChCoupling -v`
Expected: FAIL — `test_write_memory_selective_sets_surprise_level` and `test_surprise_level_reset_after_write` fail because `write_memory_selective` doesn't reference `surprise_level` yet

- [ ] **Step 3: Replace write_memory_selective with ACh-aware version**

In `scripts/train_cellmem_qwen.py`, replace the entire `write_memory_selective` method (lines 186-219) with:

```python
    def write_memory_selective(self, text, top_k=8):
        """Write only the top-k most surprising token hidden states.
        Sets surprise_level for ACh coupling (suppresses reads during encoding)."""
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = tokens["input_ids"]

        with torch.no_grad():
            outputs = self.base_model(**tokens, output_hidden_states=True)
            logits = outputs.logits

        hidden = outputs.hidden_states[-1][0]  # [T, C]

        # Compute per-token surprise
        if input_ids.size(1) > 1:
            pred_logits = logits[:, :-1, :]
            targets = input_ids[:, 1:]
            surprises = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )

            # Set ACh surprise level (used by _cross_attend_to_memory to suppress reads)
            max_surprise = surprises.max().item()
            self.surprise_level = min(1.0, max(0.0, (max_surprise - 2.0) / 4.0))

            # Select top-k most surprising positions
            k = min(top_k, surprises.size(0))
            _, top_indices = surprises.topk(k)

            if self.memory_vectors is None:
                self.memory_vectors = torch.zeros(self.n_slots, self.hidden_size,
                                                   device=self.device,
                                                   dtype=hidden.dtype)

            for idx in top_indices:
                pos = idx.item() + 1  # +1 because surprise is for predicting next token
                if pos < hidden.size(0) and self.memory_count < self.n_slots:
                    self.memory_vectors[self.memory_count] = hidden[pos].detach()
                    self.memory_count += 1

        # Reset surprise level after write phase
        self.surprise_level = 0.0
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestAChCoupling -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_cellmem_qwen.py tests/test_cellmem_wrapper.py
git commit -m "feat: ACh read/write coupling — set surprise_level during writes"
```

---

### Task 7: Conversational training data

Multi-turn conversations where some turns need memory and others don't.

**Files:**
- Modify: `scripts/train_cellmem_qwen.py` (add `_generate_conversations`)
- Modify: `tests/test_cellmem_wrapper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cellmem_wrapper.py — add at bottom

class TestConversationalData:
    """Multi-turn conversation data for training."""

    def test_conversation_has_turns(self):
        from scripts.train_cellmem_qwen import _generate_conversations
        convos = _generate_conversations(n=10, seed=42)
        assert len(convos) > 0
        for c in convos:
            assert "turns" in c
            assert len(c["turns"]) >= 3
            for turn in c["turns"]:
                assert "role" in turn  # "user" or "assistant"
                assert "content" in turn
                assert "needs_memory" in turn  # bool

    def test_mix_of_memory_and_no_memory_turns(self):
        from scripts.train_cellmem_qwen import _generate_conversations
        convos = _generate_conversations(n=20, seed=42)
        has_memory_turns = 0
        no_memory_turns = 0
        for c in convos:
            for turn in c["turns"]:
                if turn["role"] == "assistant":
                    if turn["needs_memory"]:
                        has_memory_turns += 1
                    else:
                        no_memory_turns += 1
        assert has_memory_turns > 0, "Need some memory-requiring turns"
        assert no_memory_turns > 0, "Need some no-memory turns"

    def test_user_turns_have_write_flag(self):
        from scripts.train_cellmem_qwen import _generate_conversations
        convos = _generate_conversations(n=10, seed=42)
        for c in convos:
            for turn in c["turns"]:
                if turn["role"] == "user":
                    assert "should_write" in turn
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestConversationalData -v`
Expected: FAIL with "cannot import name '_generate_conversations'"

- [ ] **Step 3: Implement _generate_conversations**

Add to `scripts/train_cellmem_qwen.py` after `_generate_mixed_data`:

```python
def _generate_conversations(n=50, seed=42):
    """Generate multi-turn conversations mixing memory-useful and memory-irrelevant turns.

    Each conversation:
    - Turn 1: User shares a fact (should_write=True)
    - Turn 2: Assistant acknowledges (needs_memory=False)
    - Turn 3: User asks something unrelated (should_write=False)
    - Turn 4: Assistant answers from general knowledge (needs_memory=False)
    - Turn 5: User asks about the original fact (should_write=False)
    - Turn 6: Assistant recalls from memory (needs_memory=True)
    """
    rng = _random.Random(seed)
    procedural = _generate_procedural_data(n * 2, seed=seed)

    chitchat_q = [
        "What's 2 + 2?", "Tell me a joke.", "What color is the sky?",
        "Name a fruit.", "What day is it?", "How are you?",
    ]
    chitchat_a = [
        "4.", "Why did the chicken cross the road? To get to the other side!",
        "The sky is blue.", "Apple.", "I'm not sure what day it is.",
        "I'm doing well, thanks!",
    ]
    ack_templates = [
        "Got it, thanks for sharing!",
        "Interesting, I'll remember that.",
        "Thanks for telling me!",
        "Noted!",
    ]

    convos = []
    for i in range(n):
        ex = procedural[i % len(procedural)]
        chitchat_idx = rng.randint(0, len(chitchat_q) - 1)

        turns = [
            {"role": "user", "content": ex["context"], "should_write": True, "needs_memory": False},
            {"role": "assistant", "content": rng.choice(ack_templates), "should_write": False, "needs_memory": False},
            {"role": "user", "content": chitchat_q[chitchat_idx], "should_write": False, "needs_memory": False},
            {"role": "assistant", "content": chitchat_a[chitchat_idx], "should_write": False, "needs_memory": False},
            {"role": "user", "content": ex["query"], "should_write": False, "needs_memory": False},
            {"role": "assistant", "content": ex["answer"], "should_write": False, "needs_memory": True},
        ]
        convos.append({"turns": turns})
    return convos
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestConversationalData -v`
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_cellmem_qwen.py tests/test_cellmem_wrapper.py
git commit -m "feat: conversational training data with memory/no-memory turns"
```

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
        for key in ["positive_recall", "negative_divergence", "poisoned_resistance",
                     "multi_memory_recall", "f1_score"]:
            assert key in doc, f"Docstring missing metric: {key}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cellmem_wrapper.py::TestEvalMetrics -v`
Expected: FAIL with "cannot import name 'eval_comprehensive'"

- [ ] **Step 3: Implement eval_comprehensive**

Add to `scripts/train_cellmem_qwen.py` after `compute_divergence_loss`:

```python
def eval_comprehensive(wrapper, tokenizer, data, device, show=3):
    """Comprehensive evaluation across positive, negative, and poisoned examples.

    Returns dict with keys:
        positive_recall: % of positive examples where answer keywords found
        negative_divergence: mean KL div between with/without memory on negatives
        poisoned_resistance: % of poisoned examples where CORRECT answer given
        multi_memory_recall: % recall with 5 accumulated memories
        f1_score: combined F1 from precision and recall metrics
    """
    stopwords = {"the", "a", "an", "is", "was", "are", "of", "in", "to", "and",
                 "that", "it", "for", "on", "with"}
    results = {"positive": [], "negative": [], "poisoned": []}

    for ex in data:
        ex_type = ex.get("type", "positive")
        if ex_type not in results:
            continue
        wrapper.clear_memory()
        wrapper.write_memory_selective(ex["context"], top_k=8)
        if wrapper.memory_count == 0:
            continue

        generated = wrapper.generate(ex["query"], max_new_tokens=32)
        answer_words = set(ex["answer"].lower().split()) - stopwords
        gen_lower = generated.lower()
        matched = sum(1 for w in answer_words if w in gen_lower)
        score = matched / max(len(answer_words), 1)
        results[ex_type].append(score > 0.5)

    pos_recall = sum(results["positive"]) / max(len(results["positive"]), 1) * 100
    poison_resist = sum(results["poisoned"]) / max(len(results["poisoned"]), 1) * 100

    # Negative divergence
    neg_divs = []
    for ex in [e for e in data if e.get("type") == "negative"][:20]:
        wrapper.clear_memory()
        wrapper.write_memory_selective(ex["context"], top_k=8)
        if wrapper.memory_count == 0:
            continue
        prompt = ex["query"] + " " + ex["answer"]
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_with = wrapper(input_ids=tokens["input_ids"])
            saved = wrapper.memory_count
            wrapper.memory_count = 0
            out_without = wrapper(input_ids=tokens["input_ids"])
            wrapper.memory_count = saved
        div = compute_divergence_loss(out_with.logits, out_without.logits).item()
        neg_divs.append(div)
    neg_div = sum(neg_divs) / max(len(neg_divs), 1)

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

    # F1
    precision = pos_recall / 100
    recall_metric = max(0, 1.0 - neg_div)
    f1 = 2 * precision * recall_metric / max(precision + recall_metric, 1e-8) * 100

    metrics = {
        "positive_recall": pos_recall,
        "negative_divergence": neg_div,
        "poisoned_resistance": poison_resist,
        "multi_memory_recall": multi_recall,
        "f1_score": f1,
    }

    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVAL")
    print(f"{'='*60}")
    for k, v in metrics.items():
        fmt = f"{v:.4f}" if "divergence" in k else f"{v:.1f}%"
        print(f"  {k:25s} {fmt}")

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
| Negative divergence | ~0.5 (high) | < 0.05 |
| Poisoned resistance | ~0% (trusts blindly) | >= 50% |
| Multi-memory recall | ~40% | >= 50% |
| F1 score | ~30% | >= 55% |
| Chat quality | degenerates | coherent |

The key metric is **negative divergence**: if this drops from ~0.5 to < 0.05, it means the gate has learned to close when memory is irrelevant — the root cause of chat degradation.
