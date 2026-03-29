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

## 4. Attention Residuals (Kimi/Moonshot, March 2026)

**Paper:** "Attention Residuals"
**Authors:** Guangyu Chen, Yu Zhang, Jianlin Su et al. (Kimi Team / Moonshot AI). arXiv:2603.15031

### Core insight
Standard residual connections accumulate all layer outputs with fixed unit weights (h_l = h_{l-1} + f_{l-1}(h_{l-1})). This causes **PreNorm dilution**: hidden-state magnitudes grow as O(L), progressively diluting each layer's relative contribution. Deeper layers must learn ever-larger outputs to remain influential.

AttnRes replaces this fixed accumulation with **learned softmax attention over depth**: each layer selectively aggregates information from all preceding layers using content-dependent weights.

### Key equations
```
h_l = α_{0→l} · h_1 + Σ_{i=1}^{l-1} α_{i→l} · f_i(h_i)           (Full AttnRes)

α_{i→l} = φ(q_l, k_i) / Σ_j φ(q_l, k_j)                          (softmax attention)

q_l = w_l  (learned pseudo-query, one d-dim vector per layer)
k_i = v_i = f_i(h_i)  (layer outputs, with RMSNorm on keys)
```

**Block AttnRes**: groups L layers into N≈8 blocks, applies full attention only over block-level representations. Reduces memory from O(Ld) to O(Nd).

### Results
- Scaling law: Block AttnRes matches baseline trained with 1.25x more compute
- 48B MoE (3B active) on 1.4T tokens: improves over baseline on all benchmarks
- GPQA-Diamond +7.5, Math +3.6, HumanEval +3.1, MMLU +1.1
- Mitigates PreNorm dilution: bounded output magnitudes, uniform gradient distribution
- Inference overhead < 2%

### Design choices relevant to CellMem v2
- **Duality of time and depth** (§6.1): residual connections compress prior information over depth, just as RNNs compress over time. AttnRes replaces the fixed depth-recurrence with attention, just as Transformers replaced RNN time-recurrence with attention. **CellMem operates on the time axis** — the two are orthogonal and complementary.
- **RMSNorm on values before attention**: prevents layers with large outputs from dominating the softmax. We should do the same on memory vectors before cross-attention read.
- **Pseudo-query init to zero**: all attention weights start uniform → no training volatility. Same philosophy as our gate init ≈ 0.
- **Input-dependent query improves** (1.731 vs 1.737) but adds sequential cost. For CellMem (64 slots, not L layers) the cost is negligible — our choice of using hidden state as query is correct.
- **Skip connection patterns** (Fig. 8): certain layers learn to attend to much earlier layers, not just the immediate predecessor. This selective long-range access over depth mirrors what CellMem does over time.

### No conflict with CellMem v2
AttnRes and CellMem operate on orthogonal axes:
- AttnRes = selective aggregation over **depth** (which layer outputs to use)
- CellMem = selective aggregation over **time** (which past experiences to recall)

A model with AttnRes could use CellMem *better*: content-dependent depth routing may handle memory-injected information more effectively than fixed residuals.

### Actionable takeaway
**RMSNorm memory vectors** before cross-attention read to prevent magnitude bias in retrieval.

---

## 5. Larimar (IBM Research, ICML 2024)

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

## 6. Neuroscience Foundation

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

## 7. Comparative Summary

```
                 Infini-attn    Memory Mosaics   Larimar         AttnRes          CellMem v2
                 (Google '24)   (Meta FAIR '25)  (IBM '24)       (Kimi '26)       (ours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Memory type      Matrix M       K/V pairs        Matrix M        Layer outputs    64 vectors
                 (d_k x d_v)    (in-context)     (K x C)         (depth cache)    in R^d_model

Axis             Time           Time             Time            Depth            Time
                 (sequence)     (sequence)       (episodic)      (layer)          (cross-session)

Write trigger    Every token    Every token      Explicit call   Every layer      Surprise > θ
                 (accumulate)   (auto)           (one-shot)      (automatic)      (anti-Hebbian)

Read mechanism   Linear attn    Softmax kernel   Least-squares   Softmax attn     Cross-attention
                                regression       projection      (1 query/layer)  + learnable gate

Forgetting       None           None             Explicit (α=-1) Softmax comp.    Decay + eviction
                                                 (surgical)      (implicit)       (passive)

Persistence      No             No               Yes             N/A (depth)      Yes
cross-session

LLM integration  In-layer       Replaces attn    External        Replaces resid.  In-layer
                 (drop-in)      (new arch)       (wrapper)       connections       (cross-attn pass)

Gate             Per-head       None             N/A             Softmax weights  Per-layer
                 scalar         (pure arch)                      (learned)        scalar

Training req.    BPTT through   End-to-end       Joint encoder-  End-to-end       Gate fine-tuning
                 segments       (standard)       memory-decoder  (standard)       (12 scalars)

Scale tested     1B-8B          8B (1T tokens)   1.3B-6B         48B (1.4T tok)   286M (toy)

Neuroscience     None           None             CLS theory      None             NIMH/Scripps
inspiration                                      (hippocampus)                    anti-Hebbian
```

### What CellMem v2 does uniquely well
1. **Surprise-gated writes** -- most selective write policy, faithful to neuroscience
2. **Cross-session persistence** -- no other approach does this
3. **Integrated in transformer** -- no external modules (vs Larimar's BERT encoder)
4. **Minimal architecture change** -- gate init ~0, model starts identical to baseline
5. **Biologically grounded** -- directly implements NIMH/Scripps anti-Hebbian principles

### Ideas to steal from the literature
1. **From Larimar:** Explicit forgetting (alpha=-1) instead of passive eviction
2. **From Infini-attention:** Delta update rule to avoid redundant writes ✅ IMPLEMENTED
3. **From Memory Mosaics:** Predictive disentanglement (value peek-ahead) for head specialization
4. **From Memory Mosaics v2:** Adaptive bandwidth for retrieval sharpness scaling
5. **From AttnRes:** RMSNorm on memory vectors before cross-attention read (prevent magnitude bias)
6. **From all:** The training problem is the real bottleneck -- need data that exercises memory retrieval

---

## 8. Neuroscience of Memory Retrieval — Mechanisms and Computational Implications

Research survey of biological memory retrieval mechanisms, conducted 2026-03-25.
Goal: identify brain-inspired gating and retrieval principles that could improve CellMem v2's read pathway.

### 8.1 Hippocampal Circuit Architecture for Retrieval

The hippocampus implements a multi-stage pipeline for memory encoding and retrieval:

```
Entorhinal Cortex (EC)
    │
    ├──→ Dentate Gyrus (DG) ──→ CA3 ──→ CA1 ──→ EC/Neocortex
    │    (pattern separation)   (pattern completion)  (output)
    │         sparse coding      recurrent attractor
    │
    └──→ CA3 (direct, perforant path)
    └──→ CA1 (direct, temporoammonic path)
```

**Key principle**: The trisynaptic pathway (EC→DG→CA3→CA1) is the *encoding* pathway. The monosynaptic pathway (EC→CA3 recurrent→CA1) is the *retrieval* pathway. Acetylcholine modulates which pathway dominates.

Sources: [Mechanisms of memory-supporting neuronal dynamics in hippocampal area CA3](https://www.cell.com/cell/fulltext/S0092-8674(24)01141-3) (Cell, 2024); [Structure and function of the hippocampal CA3 module](https://www.pnas.org/doi/10.1073/pnas.2312281120) (PNAS, 2023)

---

### 8.2 Pattern Completion in CA3 (Autoassociative Attractor Network)

**Biological mechanism**: CA3 pyramidal cells form dense recurrent connections (~2% connectivity rate, higher than previously assumed). During retrieval, a partial cue activates a subset of CA3 neurons; recurrent excitation amplifies activity until the full stored pattern is reconstructed (attractor dynamics). The network settles into the nearest stored attractor state.

**Key findings (PNAS 2023)**: 3D electron microscopy reconstruction of CA3 modules revealed connectivity rates significantly higher than previously assumed. Mathematical modeling showed these networks robustly generate pattern completion and can replay memory sequences.

**Computational model**:
```
Pattern completion = attractor dynamics in recurrent network:
  h(t+1) = f( W_recurrent * h(t) + W_input * x_cue )

where W_recurrent encodes stored patterns via Hebbian-like learning:
  W = (1/P) * Σ_p (ξ_p * ξ_p^T)    (P stored patterns, ξ = pattern vector)

Retrieval: present partial cue x_cue, iterate until convergence to attractor.
Storage capacity: ~0.14 * N patterns (Hopfield limit) for N neurons.
```

**AI implication**: CellMem's 64-slot memory bank operates as a small external attractor network. Cross-attention retrieval already implements a soft version of pattern completion (query = partial cue, keys = stored patterns, output = weighted reconstruction). However, CellMem lacks *recurrent* dynamics in retrieval -- a single cross-attention pass may not fully reconstruct complex memories. **Consider**: iterative refinement or multi-hop attention over memory slots.

Sources: [Structure and function of the hippocampal CA3 module](https://www.pnas.org/doi/10.1073/pnas.2312281120) (PNAS, 2023); [CA3 Retrieves Coherent Representations from Degraded Input](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3904133/) (Neuron, 2014)

---

### 8.3 Cholinergic Modulation: The Encoding/Retrieval Switch

**Biological mechanism**: Acetylcholine (ACh) acts as a global neuromodulator that switches the hippocampus between two modes:

| Parameter | High ACh (ENCODING) | Low ACh (RETRIEVAL) |
|-----------|---------------------|---------------------|
| CA3 recurrent synapses | Suppressed by ~85% | Active (full strength) |
| Mossy fiber (DG→CA3) | Enhanced by ~49% | Normal |
| Perforant path (EC→CA3) | Suppressed by ~50% | Active |
| Dominant pathway | Trisynaptic (EC→DG→CA3→CA1) | Monosynaptic (CA3 recurrent→CA1) |
| Mode | Pattern separation, new traces | Pattern completion, retrieval |

**The mismatch-detection trigger**: When input from EC does not match the pattern predicted by CA3 retrieval (mismatch/novelty), excitatory signals to the medial septum trigger ACh release, pushing the system into encoding mode. When input matches prediction (familiarity), ACh drops and retrieval dominates.

**Computational model (Hasselmo)**:
```
Effective CA3 recurrent weight during retrieval:
  W_eff = (1 - ACh_level) * W_recurrent

During encoding (high ACh ≈ 0.85):
  W_eff = 0.15 * W_recurrent    → recurrent retrieval suppressed
  New pattern written via DG→CA3 mossy fibers (enhanced)

During retrieval (low ACh ≈ 0):
  W_eff = 1.0 * W_recurrent     → full pattern completion
  Retrieval via CA3 recurrent + Schaffer collateral → CA1
```

**Stabilization mechanisms** (Cerebral Cortex 2022): Three mechanisms prevent runaway excitation during encoding/retrieval transitions:
1. **Short-term synaptic depression** at CA3 recurrent synapses (controls retrieval bursts)
2. **OLM interneuron inhibition** (controls encoding bursts from mossy fibers)
3. **Basket cell inhibition** (insufficient alone, needs the above two)

**AI implication**: CellMem v2 currently has a single gate that blends memory output with local attention. This is a *retrieval gate*, not an encoding/retrieval *switch*. The biological model suggests:
- **Write gate should be inversely coupled with read gate**: when surprise is high (= novelty = high ACh equivalent), *suppress retrieval and enhance encoding*. Currently our surprise threshold only controls writes; it should also *attenuate memory reads* during high-surprise moments to prevent interference from old memories during novel input processing.
- **Mismatch detection** as the trigger: compute prediction error between current hidden state and best-matching memory slot. High error → write mode, low error → read mode.

Sources: [Intrinsic Mechanisms Stabilize Encoding and Retrieval Circuits](https://pmc.ncbi.nlm.nih.gov/articles/PMC9121438/) (Cerebral Cortex, 2022); [Evidence for Encoding versus Retrieval Scheduling by Theta Phase and Acetylcholine](https://www.jneurosci.org/content/33/20/8689) (J. Neuroscience, 2013); [The Role of Acetylcholine in Learning and Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC2659740/) (Current Opinion in Neurobiology, 2006)

---

### 8.4 CA3→CA1 Schaffer Collateral: The Retrieval Output Pathway

**Biological mechanism**: Retrieved patterns from CA3 are transmitted to CA1 via Schaffer collateral axons. CA1 acts as a *comparator*: it receives both the retrieved pattern (from CA3 via Schaffer collateral) and the current sensory input (from EC via temporoammonic pathway). The comparison determines whether the memory matches current context.

**Gating by inhibition**: Schaffer collateral inputs to CA1 follow different connectivity rules for excitatory vs inhibitory neurons:
- **Pyramidal cells**: Non-random connectivity (structured, memory-specific)
- **PV+ interneurons**: More random connectivity (provides broad inhibition)

This means PV+ interneurons provide a *baseline inhibitory tone* that the specific excitatory Schaffer inputs must overcome. Only strongly activated memories (good pattern completion) produce enough excitation to surpass inhibition and reach CA1 output neurons.

**Somatostatin interneuron metaplasticity**: SOM+ interneurons differentially regulate LTP at the two CA1 input pathways:
- **Schaffer collateral (retrieval)**: LTP facilitated
- **Temporoammonic (direct sensory)**: LTP reduced

This creates a *retrieval bias* -- the system preferentially strengthens memory-based inputs over direct sensory inputs in CA1.

**AI implication**: The biological CA1 comparator suggests CellMem should *compare* memory retrieval output with the current hidden state before gating. Currently the gate is a simple learned scalar. A **content-dependent gate** that compares memory readout quality with local context could be more effective:
```
gate = sigmoid(MLP([h_local; h_memory; h_local - h_memory]))
```

Sources: [Schaffer Collateral Inputs to CA1 Follow Different Connectivity Rules](https://www.jneurosci.org/content/38/22/5140) (J. Neuroscience, 2018); [Perforant pathway and CA3-Schaffer collateral coordinate spatial learning](https://www.nature.com/articles/s42003-026-09577-z) (Comm. Biology, 2026)

---

### 8.5 Sharp-Wave Ripples: Memory Replay and Consolidation

**Biological mechanism**: During quiet wakefulness and sleep, CA3 generates spontaneous bursts of activity called sharp-wave ripples (SWRs, 150-250 Hz). These events replay stored memory sequences at 15-20x speed, propagating from hippocampus to neocortex for consolidation.

**Key 2024 findings** (Science):
- SWRs during waking *select* which experiences to consolidate. Not all events are replayed -- SWRs occur preferentially during reward consumption and at decision points
- The spike content of waking SWRs decoded specific trial blocks that were later replayed during sleep SWRs
- This constitutes a **neurophysiological tagging mechanism**: waking SWRs "tag" important events, and sleep SWRs consolidate the tagged ones

**Large SWRs drive consolidation** (Neuron, 2025): Only a subset of *large-amplitude* SWRs are associated with hippocampo-cortical memory reactivation. Optogenetic boosting of SWRs during post-task sleep enhanced ensemble reactivation in hippocampus and prefrontal cortex, improving memory performance.

**Replay without ripples** (Nature Comm., 2025): Replay can occur without ripples, suggesting they are distinct but coordinated processes. Ripples selectively tag a subset of replays linked to learning or novelty.

**Computational model (eLife 2022)**: A spiking network of 8000 CA3 pyramidal cells + 150 PV+ interneurons produced both forward and backward sequence replay during spontaneous bursts. The key ingredients were:
1. **Symmetric STDP rule** (not asymmetric) → creates chain-like connectivity
2. **Cellular adaptation** (intrinsic neuronal properties regulate excitability)
3. **Structured recurrent weights**: few strong synapses near the diagonal (overlapping place fields) embedded in a background of weak connections

```
Weight structure after learning:
  W_ij ~ strong  if |place_field_i - place_field_j| < threshold
  W_ij ~ weak    otherwise

Replay: spontaneous activation of one cell in the chain triggers
        sequential reactivation, producing time-compressed replay.
```

**AI implication**: CellMem currently writes memories but has no *replay/consolidation* mechanism. The biology suggests:
- **Offline consolidation**: During periods of low activity or at sequence boundaries, replay stored memories to strengthen neocortical representations (= fine-tune the base model on memory content)
- **Selective replay**: Only replay high-surprise (tagged) memories, not everything
- **Compressed replay**: Replay at accelerated timescale (shorter sequences of memory content)
- **For CellMem v2 specifically**: Consider a "memory rehearsal" step at segment boundaries where stored memory vectors are fed back through the network to update the model's understanding

Sources: [Selection of experience for memory by hippocampal sharp wave ripples](https://www.science.org/doi/10.1126/science.adk8261) (Science, 2024); [Large sharp-wave ripples promote hippocampo-cortical memory reactivation](https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1) (Neuron, 2025); [SWR and sequence replay emerge from structured synaptic interactions in CA3](https://elifesciences.org/articles/71850) (eLife, 2022); [Replay without sharp wave ripples](https://www.nature.com/articles/s41467-025-65181-5) (Nature Comm., 2025)

---

### 8.6 BTSP: One-Shot Memory Encoding for Content-Addressable Retrieval

**Biological mechanism**: Behavioral Timescale Synaptic Plasticity (BTSP) is a recently discovered non-Hebbian plasticity rule in hippocampal CA1 and CA3. Unlike classical STDP (millisecond timescale), BTSP operates over *seconds* and is triggered by a single dendritic plateau potential. It is the mechanism by which a previously silent neuron can become a place cell in one trial.

**BTSP properties**:
- Triggered by single plateau potential (not repeated co-activation)
- Bidirectional (both potentiation and depression)
- Operates on behavioral timescale (seconds, not milliseconds)
- One-shot (single event sufficient for memory trace creation)
- More frequent during novel experiences, decays with familiarity

**Computational model (Nature Comm., 2024 / PLOS Comp. Bio., 2023)**:
```
BTSP weight update rule:
  w_ij^(k) = w_ij^(k-1) + Δw_ij

  Δw_ij depends on timing between:
    - presynaptic activity of cell j
    - plateau potential in postsynaptic cell i

  Plasticity windows: f_P(θ) for potentiation, f_D(θ) for depression
  where θ = spatial/temporal offset between pre-activity and plateau

After learning, network supports bump attractors:
  - Localized activity patterns representing stored memories
  - Pattern completion from partial cues via attractor dynamics

Memory capacity: η_cr ~ M² log(N)
  where M = population redundancy, N = total neurons
  (scales quadratically with redundancy under optimal sparseness)
```

**Content-addressable memory with binary synapses**: A 2024 Nature Communications paper showed BTSP can create a functionally powerful content-addressable memory even with binary (0/1) synaptic weights. The model also reproduces the *repulsion effect* of human memory (similar traces are pushed apart for better discrimination).

**CA3-specific BTSP (Cell, 2024)**: BTSP at recurrent CA3 synapses produces attractor dynamics under online learning conditions. This means CA3 simultaneously learns and retrieves -- it does not need separate training and inference phases.

**AI implication**: BTSP validates CellMem's core design:
- **One-shot writes** (surprise-gated, no gradient) ↔ BTSP plateau potential
- **Write on novelty** (high surprise) ↔ BTSP more frequent in novel environments
- **Bidirectional plasticity** ↔ CellMem could implement both write and anti-Hebbian erasure
- **Content-addressable retrieval** ↔ Cross-attention read
- **Binary synapses suffice** → suggests memory vectors don't need full float32 precision; quantized memory banks may work

**New idea from BTSP**: The *repulsion effect* (pushing similar memories apart) could be implemented as a decorrelation loss on memory slots, preventing redundant storage and improving discriminability.

Sources: [A simple model for BTSP provides content addressable memory with binary synapses](https://www.nature.com/articles/s41467-024-55563-6) (Nature Comm., 2024); [Rapid memory encoding with BTSP in a recurrent network](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011139) (PLOS Comp. Bio., 2023); [Behavioral timescale synaptic plasticity properties and functions](https://www.nature.com/articles/s41593-026-02214-2) (Nature Neuroscience, 2026); [Mechanisms of memory-supporting neuronal dynamics in CA3](https://www.cell.com/cell/fulltext/S0092-8674(24)01141-3) (Cell, 2024)

---

### 8.7 Prefrontal Cortex: Top-Down Gating of Memory Retrieval

**Biological mechanism**: The prefrontal cortex (PFC) exerts top-down control over hippocampal memory retrieval, selecting which memories are accessed based on current goals and task demands.

**Gating mechanisms**:
1. **Frontostriatal gating**: PFC→striatum→thalamus circuit gates what enters and exits working memory. Dopamine signals in striatum act as "open/close" gate signals.
2. **Direct PFC→hippocampus inhibition**: Right anterior DLPFC and VLPFC can suppress hippocampal retrieval (thought suppression, directed forgetting).
3. **Attentional templates**: PFC neurons encode target-associated information that biases hippocampal retrieval toward goal-relevant memories.

**SOM+ interneuron gating (PNAS, 2024)**: During memory consolidation, somatostatin-positive interneurons in neocortex act as "gatekeepers" for hippocampal-to-cortical information transfer:
```
Disinhibition circuit:
  Hippocampal input → activates SOM+ interneurons
  SOM+ interneurons → inhibit PV+ interneurons
  PV+ interneurons (now suppressed) → release pyramidal cells from inhibition
  Result: hippocampal input gates through to neocortical pyramidal cells

  HPC → SOM+ ⊣ PV+ ⊣ Pyramidal → disinhibition → memory transfer
```

**VIP interneurons and novelty (2024)**: VIP interneurons in CA1 become highly active during novel environments while SOM+ and PV+ populations show inverse activity. This VIP-mediated disinhibition constitutes rapid encoding of novelty and acquisition of recognition memory.

**Working memory gating model (Neural Computation, 2021)**:
```
Selective gating via attentional tagging:
  - Input gate:  g_in = f(PFC_state, input_novelty)
  - Output gate: g_out = f(PFC_state, task_relevance)
  - Maintenance: stable attractors hold information when gates are closed

The gate is NOT a simple scalar -- it is content-dependent and
task-dependent, modulated by PFC representations of current goals.
```

**AI implication**: CellMem's per-layer scalar gate is biologically impoverished compared to the brain's multi-level gating system. The biology suggests:
- **Task-dependent retrieval**: The gate should be conditioned on a "task representation" (e.g., system prompt embedding), not just learned as a fixed scalar
- **Disinhibition for write**: Novel input should *actively suppress* the retrieval pathway (not just open the write pathway independently)
- **Content-dependent output gating**: Which memory slots contribute to output should depend on current task/query relevance, not just key-query similarity

Sources: [Learning-dependent gating of hippocampal inputs by frontal interneurons](https://www.pnas.org/doi/10.1073/pnas.2403325121) (PNAS, 2024); [Flexible Working Memory Through Selective Gating and Attentional Tagging](https://direct.mit.edu/neco/article/33/1/1/95670/) (Neural Computation, 2021); [Hippocampal GABAergic interneurons and memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC10593603/) (Neuron, 2023); [The role of inhibitory circuits in hippocampal memory processing](https://www.nature.com/articles/s41583-022-00599-0) (Nature Reviews Neuroscience, 2022)

---

### 8.8 Complementary Learning Systems (CLS) Theory

**Core theory (McClelland, McNaughton, O'Reilly, 1995; updated Kumaran, Hassabis, McClelland, 2016)**:

Intelligent agents need two complementary learning systems:

| System | Brain region | Learning rate | Representations | Function |
|--------|-------------|---------------|-----------------|----------|
| Slow | Neocortex | Low | Overlapping, distributed | Extract statistical structure, generalize |
| Fast | Hippocampus | High | Sparse, separated | Encode specific episodes, avoid interference |

**The catastrophic interference problem**: If a single network learns new information too fast, it overwrites old knowledge. CLS solves this with a two-system architecture where the hippocampus rapidly stores new episodes and *replays* them to the neocortex, interleaved with ongoing experience, enabling gradual integration without catastrophic forgetting.

**2016 update key additions**:
1. **Replay serves goal-dependent weighting**: Not all memories are replayed equally; reward-relevant experiences are preferentially replayed
2. **Hippocampus can generalize**: Recurrent activation of hippocampal traces can support some forms of generalization (not just episodic specifics)
3. **Neocortex can learn fast**: When new information is *consistent* with existing structure, neocortical learning can be rapid (no catastrophic interference if compatible)

**Hippocampal Memory Indexing Theory** (Teyler & DiScenna, 1986; updated 2024):
The hippocampus stores an *index* -- pointers to neocortical patterns, not the patterns themselves. During retrieval, the hippocampal index reactivates the original neocortical ensemble. This is analogous to storing keys (hippocampus) that retrieve values (neocortex).

**Memory consolidation as replay**:
```
During sleep / quiet wakefulness:
  1. CA3 spontaneously reactivates stored pattern (SWR)
  2. Pattern propagates to CA1 → neocortex
  3. Neocortical synapses are slightly strengthened
  4. Repeated over days/weeks → gradual transfer
  5. Eventually, neocortex can retrieve without hippocampus

This is functionally equivalent to:
  - Knowledge distillation (teacher=hippocampus, student=neocortex)
  - Experience replay in reinforcement learning (DQN buffer)
```

**AI implication**: CellMem v2 IS a CLS implementation:
- **Base transformer** = neocortex (slow learning via pretraining)
- **Memory bank** = hippocampus (fast one-shot writes at inference)
- **Cross-attention read** = hippocampal retrieval biasing neocortical processing
- **Missing piece**: **Replay/consolidation** -- periodically fine-tune the base model on memory bank contents to transfer episodic knowledge into parametric knowledge

The CLS framework also suggests that CellMem's gate initialization at ~0 is correct: the "neocortex" (transformer) should dominate initially, with the "hippocampus" (memory) gradually gaining influence as useful memories accumulate.

Sources: [What Learning Systems do Intelligent Agents Need? CLS Updated](https://pubmed.ncbi.nlm.nih.gov/27315762/) (Trends in Cognitive Sciences, 2016); [Why there are CLS in hippocampus and neocortex](https://pubmed.ncbi.nlm.nih.gov/7624455/) (Psych. Review, 1995); [Hippocampal memory indexing theory](https://pubmed.ncbi.nlm.nih.gov/3008780/) (Hippocampus, 1986); [Memory consolidation from a reinforcement learning perspective](https://public-pages-files-2025.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1538741/pdf) (Frontiers Comp. Neuro., 2024)

---

### 8.9 Dentate Gyrus: Pattern Separation as the Encoding Gate

**Biological mechanism**: Before a memory reaches CA3 for storage, the dentate gyrus (DG) transforms overlapping input patterns from the entorhinal cortex into sparse, decorrelated representations. This is *pattern separation* -- the computational opposite of CA3's pattern completion.

**Key properties**:
- DG has ~1 million granule cells (10x more than CA3 pyramidal cells)
- Only ~2-5% of granule cells are active at any time (extreme sparsity)
- Each granule cell contacts CA3 via "detonator" mossy fiber synapses (very strong, few connections)
- This sparse-to-strong architecture ensures that similar EC inputs activate *different* CA3 ensembles

**Computational model**:
```
Pattern separation via competitive inhibition:
  DG sparsity enforced by strong feedback inhibition from basket cells.

  For input patterns x1, x2 with overlap cos(x1, x2) = 0.8:
    DG output: cos(DG(x1), DG(x2)) ≈ 0.2     (decorrelated)

  This prevents CA3 from confusing similar memories.

  Sparse coding: y_i = f(W_EC→DG * x - threshold)
  where threshold is set by inhibitory interneurons to maintain ~2-5% activity

  The DG→CA3 mossy fiber "detonator" synapses are so strong that
  a single active granule cell can fire a CA3 pyramidal cell,
  forcing a new CA3 representation even if recurrent CA3 weights
  would otherwise pull toward an existing attractor.
```

**Frequency-dependent separation (2024)**: A new computational model shows DG granule cells perform frequency-dependent pattern separation, with a U-shaped relationship between input oscillation frequency and separation quality. Dendritic properties contribute to sparsity control.

**AI implication**: CellMem v2 has no explicit pattern separation before writing. If two similar inputs both exceed the surprise threshold, they may write nearly identical vectors to different memory slots (redundancy). The DG model suggests:
- **Decorrelation before write**: Apply a sparsification or orthogonalization step to the memory vector before storing it
- **Competitive inhibition**: Only write if the new vector is sufficiently different from existing slots (minimum cosine distance threshold)
- **The repulsion effect from BTSP** (Section 8.6) is the plasticity-level complement to DG's circuit-level separation

Sources: [A Combinatorial Model for Dentate Gyrus Sparse Coding](https://www.osti.gov/biblio/1371475) (Neural Computation, 2017); [Granule cells perform frequency-dependent pattern separation](https://pubmed.ncbi.nlm.nih.gov/37950569/) (Hippocampus, 2024); [Dendrites of DG granule cells contribute to pattern separation by controlling sparsity](https://pmc.ncbi.nlm.nih.gov/articles/PMC5217096/) (Hippocampus, 2017)

---

### 8.10 Summary: Biological Principles for CellMem v2 Retrieval

```
BRAIN MECHANISM                          CELLMEM v2 STATUS        ACTIONABLE IDEA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ACh encoding/retrieval switch         Partial (write gate)     Couple write & read gates inversely
                                                                  (high surprise → suppress reads)

2. CA3 attractor pattern completion      Partial (cross-attn)     Consider multi-hop retrieval
                                                                  or iterative refinement

3. DG pattern separation before write    Missing                  Decorrelate/orthogonalize before
                                                                  storing; min cosine distance check

4. CA1 comparator (memory vs sensory)    Missing                  Content-dependent gate comparing
                                                                  h_local vs h_memory

5. SWR memory replay/consolidation       Missing                  Periodic replay of high-surprise
                                                                  memories for model fine-tuning

6. BTSP one-shot bidirectional writes    Implemented (surprise)   Add repulsion effect (push similar
                                                                  memories apart); try quantized memory

7. PFC top-down retrieval gating         Missing (scalar gate)    Task-conditioned gate (system prompt
                                                                  embedding modulates retrieval)

8. Disinhibition circuits (VIP/SOM/PV)   Missing                  Novelty-triggered write pathway
                                                                  that actively suppresses retrieval

9. CLS two-system architecture           Implemented              Add replay/consolidation to transfer
                                                                  episodic → parametric knowledge

10. Hippocampal indexing (keys→cortex)   Partial                  Memory stores indices/keys, not full
                                                                  representations (saves space)
```

**Priority ranking for CellMem v2 implementation**:
1. **Inverse coupling of read/write gates** (ACh switch) -- simplest, highest impact
2. **Content-dependent gate** (CA1 comparator) -- replaces scalar with adaptive gate
3. **Decorrelation before write** (DG pattern separation) -- prevents slot redundancy
4. **Task-conditioned retrieval** (PFC gating) -- if system prompts vary
5. **Memory replay** (SWR consolidation) -- long-term, requires training infrastructure

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
- Chen, Zhang, Su et al. "Attention Residuals." arXiv:2603.15031 (Kimi/Moonshot AI, 2026)
- Memory-Augmented Transformers Systematic Review. arXiv:2508.10824 (2025)
- Mechanisms of memory-supporting neuronal dynamics in hippocampal area CA3. Cell (2024). S0092-8674(24)01141-3
- Structure and function of the hippocampal CA3 module. PNAS (2023). doi:10.1073/pnas.2312281120
- Intrinsic Mechanisms Stabilize Encoding and Retrieval Circuits in a Hippocampal Network Model. Cerebral Cortex (2022). PMC9121438
- Hasselmo. Evidence for Encoding vs Retrieval Scheduling by Theta Phase and ACh. J. Neuroscience 33(20):8689 (2013)
- Hasselmo. The Role of Acetylcholine in Learning and Memory. Current Opinion in Neurobiology (2006). PMC2659740
- Schaffer Collateral Inputs to CA1 Follow Different Connectivity Rules. J. Neuroscience 38(22):5140 (2018)
- Selection of experience for memory by hippocampal sharp wave ripples. Science (2024). doi:10.1126/science.adk8261
- Large sharp-wave ripples promote hippocampo-cortical memory reactivation. Neuron (2025). S0896-6273(25)00756-1
- SWR and sequence replay emerge from structured synaptic interactions in CA3. eLife (2022). doi:10.7554/eLife.71850
- Replay without sharp wave ripples in a spatial memory task. Nature Comm. (2025). doi:10.1038/s41467-025-65181-5
- A simple model for BTSP provides content addressable memory with binary synapses. Nature Comm. (2024). doi:10.1038/s41467-024-55563-6
- Rapid memory encoding with BTSP in a recurrent network. PLOS Comp. Bio. (2023). doi:10.1371/journal.pcbi.1011139
- Behavioral timescale synaptic plasticity: properties, elements and functions. Nature Neuroscience (2026). doi:10.1038/s41593-026-02214-2
- Learning-dependent gating of hippocampal inputs by frontal interneurons. PNAS (2024). doi:10.1073/pnas.2403325121
- Flexible Working Memory Through Selective Gating and Attentional Tagging. Neural Computation 33(1):1 (2021)
- Hippocampal GABAergic interneurons and memory. Neuron (2023). S0896-6273(23)00475-0
- The role of inhibitory circuits in hippocampal memory processing. Nature Reviews Neuroscience (2022). doi:10.1038/s41583-022-00599-0
- Kumaran, Hassabis, McClelland. What Learning Systems do Intelligent Agents Need? CLS Updated. Trends Cogn. Sci. (2016)
- McClelland, McNaughton, O'Reilly. Why there are CLS in hippocampus and neocortex. Psych. Review (1995)
- Teyler & DiScenna. The hippocampal memory indexing theory. Behavioral Neuroscience (1986)
- Memory consolidation from a reinforcement learning perspective. Frontiers Comp. Neuroscience (2024). doi:10.3389/fncom.2024.1538741
- A Combinatorial Model for Dentate Gyrus Sparse Coding. Neural Computation (2017). doi:10.1162/NECO_a_00905
- Granule cells perform frequency-dependent pattern separation. Hippocampus (2024). PMID:37950569
- Bio-inspired computational memory model of the Hippocampus (spiking CAM). Neural Networks (2024). S0893-6080(24)00398-8
- Perforant pathway and CA3-Schaffer collateral coordinate spatial learning. Comm. Biology (2026). doi:10.1038/s42003-026-09577-z
