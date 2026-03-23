# CellMem v2 — Literature Review

Research survey of memory-augmented transformer architectures, conducted 2026-03-23.
Goal: understand the landscape, validate CellMem v2 design choices, identify improvements.

---

## 1. Why Memory-Augmented Transformers Haven't Taken Off

### The brute-force context window won (for now)
FlashAttention + KV-cache + Ring Attention have pushed vanilla transformers to 128K-2M tokens without architectural changes. This is simpler to engineer and "good enough" for most production cases. Companies (Google, Anthropic, OpenAI) have invested heavily in this path because it requires no changes to the training recipe.

### Training data doesn't teach memory use
HuggingFace attempted to replicate Infini-attention and failed. Root cause: training data consists of short documents, so the model never encounters examples requiring cross-segment retrieval. It never learns *when* to use memory because it never *needs* memory during training. This is a vicious cycle.

### RAG absorbed the practical demand
For applications needing "long-term memory," RAG with vector databases is much simpler: frozen model + external retrieval. This siphoned practical interest away from architectural memory solutions.

### Additional barriers
- **Staleness**: Cached representations become misaligned when model weights are updated (fine-tuning)
- **Compression degrades with scale**: Fixed-size memory = information bottleneck
- **No forgetting mechanism**: Old bindings interfere with new ones (catastrophic interference)
- **Retrieval latency**: kNN-based approaches add 100-200ms per turn
- **Ecosystem lock-in**: Hardware kernels, serving infra, RLHF pipelines all optimized for standard transformers

### The future direction
The consensus is shifting toward:
- **SSM-Transformer hybrids** (Mamba, Nemotron-H) with implicit recurrent memory
- **Hierarchical multi-timescale memory** inspired by neuroscience
- **Context engineering** (curate 8K-32K well) rather than architectural changes
- **Memory Mosaics** as a principled alternative at scale

Sources: HuggingFace blog (Infini-attention replication failure), IBM Research, Memory-Augmented Transformers Systematic Review (Aug 2025, arXiv:2508.10824)

---

## 2. Infini-attention (Google, April 2024)

**Paper:** "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
**Authors:** Munkhdalai, Faruqui, Gopal (Google). arXiv:2404.07143

### Architecture
Drop-in replacement for standard MHA. Each layer combines:
1. **Local causal attention** (standard softmax over current segment of N tokens)
2. **Compressive memory** M in R^{d_key x d_value} (fixed-size associative matrix, accumulates across segments via linear attention)
3. **Learned gate** (scalar per head, blends local + memory output)

### Key equations
```
Memory retrieval:   A_mem = sigma(Q) M_{s-1} / (sigma(Q) z_{s-1})     where sigma = ELU+1
Memory update:      M_s = M_{s-1} + sigma(K)^T V                      (linear variant)
Delta update:       M_s = M_{s-1} + sigma(K)^T (V - sigma(K)M_{s-1}/...) (delta variant)
Gating:             A = sigmoid(beta) * A_mem + (1-sigmoid(beta)) * A_dot
```

### Results
- 114x compression ratio vs Memorizing Transformers
- Perplexity 9.65 on PG19 (vs 11.37 Memorizing Transformers, 11.88 Transformer-XL)
- Solved 1M token passkey retrieval (trained on 5K, generalized to 1M = 200x extrapolation)
- New SOTA on 500K BookSum summarization

### Design choices shared with CellMem v2
- No positional encoding on memory (correct: memories have no position)
- Same W_k/W_v projections for both token and memory attention
- Per-head scalar gate (we use per-layer)
- Gate initialized near 0

### Key differences from CellMem v2
- **Linear attention** for memory retrieval (vs our softmax cross-attention)
- **Accumulates everything** (no surprise gating, writes all tokens)
- **No persistence** across sessions (resets per sequence)
- **No forgetting** (delta rule avoids redundancy but doesn't remove stale info)
- **Per-head gate** (vs our per-layer gate)

### Limitations
- Lossy compression (fixed-size matrix for infinite history)
- Linear attention is weaker than softmax for fine-grained retrieval
- Requires BPTT through segments (memory-intensive, vanishing gradients)
- Per-head gate, not per-token (can't dynamically decide per position)

---

## 3. Memory Mosaics (Meta FAIR, ICLR 2025)

**Paper:** "Memory Mosaics" by Zhang, Nolte, Sadhukhan, Chen, Bottou (FAIR/Meta, CMU, NYU)
arXiv:2405.06394. Follow-up at scale: arXiv:2507.03285 (NeurIPS 2025 Oral)

### Core insight
"Everything is associative memory." The transformer is reframed as a network of memory units:
- **Contextual Memory Units** replace self-attention (dynamic, fill during inference)
- **Persistent Memory Units** replace FFN/MLP (fixed, learned during training)

Self-attention IS a special case of kernel regression on an associative memory.

### Architecture
Each block has 12 contextual + 12 persistent memory units.

**Read/Write (kernel regression / Nadaraya-Watson):**
```
y_t = sum_i [exp(beta * k_t^T k_i) / sum_j exp(beta * k_t^T k_j)] * v_i
```

**The "peek ahead" trick (predictive disentanglement):**
- Keys k_t = f(x_t, x_{t-1}, ...) -- based on past only
- Values v_t = g(x_{t+1}, x_t, x_{t-1}, ...) -- can see ONE token ahead
- This asymmetry drives automatic task decomposition: each head specializes in predicting a different aspect of the next token

**Leaky averaging replaces positional encoding:**
```
k_bar_t = k_tilde_t + lambda * k_bar_{t-1}
```

### Results at scale (v2, 8B, 1T tokens)
- Training knowledge: identical to transformers (52.2%)
- New-knowledge at 4K context: 59.3% vs 57.7% (transformer)
- New-knowledge at 32K context: 53.4% vs 41.1% (12.3 point gap!)
- Context extrapolation: trained on 4K, works at 32K without fine-tuning
- Data efficiency: MM v2 on 1T tokens > transformer on 8T tokens

### v2 refinements
- Adaptive bandwidth: beta = beta_1 * n^alpha + beta_0 (learnable)
- Gated key extraction (input-dependent, replaces fixed leaky averaging)
- Three-level hierarchy: short-term + long-term + persistent memory

### Relevance to CellMem v2
- Confirms that memory integration in transformers works at scale (8B)
- Predictive disentanglement (value peek-ahead) is a powerful idea we don't use
- Adaptive bandwidth could improve our retrieval quality
- At scale, persistent memory reverts to standard SwiGLU FFN (purity vs practicality)

### Limitations
- Quadratic complexity (no FlashAttention equivalent)
- 13% more compute than transformer at 8B scale
- No persistence across sessions
- No explicit forgetting

---

## 4. Larimar (IBM Research, ICML 2024)

**Paper:** "Large Language Models with Episodic Memory Control"
**Authors:** Das, Chaudhury, Nelson, Melnyk et al. (IBM + Princeton). arXiv:2403.11901

### Architecture
External episodic memory module wrapping a frozen LLM. Inspired by Complementary Learning Systems (CLS) theory:
- LLM = neocortex (slow semantic knowledge)
- Memory = hippocampus (fast one-shot episodic learning)

Components:
- **Encoder** (BERT-large): input -> latent space R^768
- **Memory M** (R^{512 x 768}): fixed-size matrix
- **Decoder** (GPT-2 1.3B or GPT-J 6B): frozen, conditioned via memory readout as KV cache

### The prediction-error update rule (key equations)
```
M_i = M_{i-1} + alpha * C_i^{-1} * W_i^T * (Z_i - W_i * M_{i-1})
                                              ╰──────────────────╯
                                               prediction error

alpha = +1  -->  WRITE (Hebbian consolidation)
alpha = -1  -->  FORGET (anti-Hebbian erasure)
```

Based on Kanerva Machine (Wu & Wayne 2018), reformulated so Bayesian updates reduce to closed-form least-squares solutions. No gradients needed at inference time.

The term (Z_i - W_i * M_{i-1}) is structurally identical to prediction-error-driven plasticity: the difference between what should be stored and what the memory already predicts.

### Results
- 4-10x faster than ROME/GRACE for knowledge editing
- 100% edit success on single facts
- 97% retention after 1000 sequential edits
- Perfect forgetting: 0% recall of forgotten fact, 99.3% retention of others
- Robust to context length: 80% recall at 64/128/256 facts (Mistral-7B drops from 98% to 42%)

### Relevance to CellMem v2
- **Confirms prediction-error-driven writes are the right principle**
- **Explicit forgetting via alpha=-1** is more surgical than our "overwrite lowest surprise" eviction
- **One-shot write** (no gradient) is similar to our inference-time writes
- **LLM remains frozen** like our base training (gates start at 0)

### Differences from CellMem v2
- Requires separate encoder (BERT) + decoder -- not integrated into the transformer
- Memory is external, not in the same embedding space as hidden states
- Limited to 64-token fact chunks (restrictive)
- Designed for knowledge editing, not general persistent memory
- No cross-attention integration (memory readout injected as KV cache)

---

## 5. Neuroscience Foundation

### NIMH/Scripps 2025: "Synaptic plasticity rules driving representational shifting in the hippocampus"
Madar, Jiang, Dong & Sheffield. Nature Neuroscience 28, 848-860 (2025).

Key findings:
1. **BTSP (Behavioral Timescale Synaptic Plasticity)**, not classical Hebbian STDP, drives hippocampal place field dynamics
2. BTSP causes **large, one-shot** changes in synaptic strength (like Larimar's one-shot writes, like CellMem's surprise-gated writes)
3. BTSP events are **more frequent during novel experiences** and decay after place field onset (= surprise-driven)
4. The plasticity is **non-Hebbian and bidirectional** (not dependent on post-synaptic firing rate)

### "A Non-Hebbian Code for Episodic Memory" (Science Advances, 2024)
Non-Hebbian plasticity (presynaptic-only) is sufficient for flexible episodic memory. Maps directly onto hippocampal mossy fiber synapses and BTSP. Supports one-shot sequential and associative recall.

### Connection to CellMem v2
CellMem v2's design principles are directly validated by this neuroscience:
- **Memory = structural change** (vectors in embedding space, not a separate database)
- **Anti-Hebbian / prediction-error-driven** (write on surprise, not on repetition)
- **During experience** (inference-time, not training-time)
- **One-shot** (single write per surprising event)
- **Decay** (biological forgetting analog)

---

## 6. Comparative Summary

```
                 Infini-attn    Memory Mosaics   Larimar         CellMem v2
                 (Google '24)   (Meta FAIR '25)  (IBM '24)       (ours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Memory type      Matrix M       K/V pairs        Matrix M        64 vectors
                 (d_k x d_v)    (in-context)     (K x C)         in R^768

Write trigger    Every token    Every token      Explicit call   Surprise > threshold
                 (accumulate)   (auto)           (one-shot)      (anti-Hebbian)

Read mechanism   Linear attn    Softmax kernel   Least-squares   Cross-attention
                                regression       projection      + learnable gate

Forgetting       None           None             Explicit (a=-1) Decay + eviction
                                                 (surgical)      (passive)

Persistence      No             No               Yes             Yes
cross-session

LLM integration  In-layer       Replaces attn    External        In-layer
                 (drop-in)      (new arch)       (wrapper)       (cross-attn pass)

Gate             Per-head       None             N/A             Per-layer
                 scalar         (pure arch)                      scalar

Training req.    BPTT through   End-to-end       Joint encoder-  Gate fine-tuning
                 segments       (standard)       memory-decoder  (12 scalars)

Scale tested     1B-8B          8B (1T tokens)   1.3B-6B         286M (toy)

Neuroscience     None           None             CLS theory      NIMH/Scripps
inspiration                                      (hippocampus)   anti-Hebbian
```

### What CellMem v2 does uniquely well
1. **Surprise-gated writes** -- most selective write policy, faithful to neuroscience
2. **Cross-session persistence** -- no other approach does this
3. **Integrated in transformer** -- no external modules (vs Larimar's BERT encoder)
4. **Minimal architecture change** -- gate init ~0, model starts identical to baseline
5. **Biologically grounded** -- directly implements NIMH/Scripps anti-Hebbian principles

### Ideas to steal from the literature
1. **From Larimar:** Explicit forgetting (alpha=-1) instead of passive eviction
2. **From Infini-attention:** Delta update rule to avoid redundant writes
3. **From Memory Mosaics:** Predictive disentanglement (value peek-ahead) for head specialization
4. **From Memory Mosaics v2:** Adaptive bandwidth for retrieval sharpness scaling
5. **From all:** The training problem is the real bottleneck -- need data that exercises memory retrieval

---

## References

- Munkhdalai et al. "Leave No Context Behind: Infini-attention." arXiv:2404.07143 (2024)
- Zhang et al. "Memory Mosaics." arXiv:2405.06394 (ICLR 2025)
- Zhang & Bottou. "Memory Mosaics at scale." arXiv:2507.03285 (NeurIPS 2025)
- Das et al. "Larimar: LLMs with Episodic Memory Control." arXiv:2403.11901 (ICML 2024)
- Wu & Wayne. "The Kanerva Machine." arXiv:1804.01756 (2018)
- Madar et al. "Synaptic plasticity rules." Nature Neuroscience 28, 848-860 (2025)
- "A non-Hebbian code for episodic memory." Science Advances (2024), doi:10.1126/sciadv.ado4112
- HuggingFace blog: "A failed experiment with Infini-Attention" (2024)
- Memory-Augmented Transformers Systematic Review. arXiv:2508.10824 (2025)
