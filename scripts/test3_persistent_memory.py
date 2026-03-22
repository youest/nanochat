"""
Test 3: Persistent Memory — does MemoryBank remember across inference invocations?

THIS IS AN INFERENCE-ONLY TEST. The model weights are FROZEN after training.
Only the MemoryBank state (slots) and CellMem state (M) persist across invocations.

Setup:
  Phase 0 (TRAIN): Train tiny transformer on a fact-recall task (all variants).
                    Then FREEZE all model weights.
  Phase 1 (LEARN): Inference-only. Feed sequences with ORIGINAL facts.
                    MemoryBank accumulates slots. CellMem M accumulates.
  Phase 2 (INTERFERE): Inference-only. Feed unrelated noise sequences.
                        This should NOT overwrite anything (weights frozen).
  Phase 3 (RECALL): Inference-only. Query original facts.
                     MemoryBank should help retrieve them.

Compare:
  - baseline: no CellMem, no MemoryBank
  - cellmem_reset: CellMem with M reset per sequence (no cross-sequence memory)
  - cellmem_persistent: CellMem with persistent M (no reset) + MemoryBank

The key metric: accuracy in Phase 3 vs Phase 1.
With frozen weights and no memory, Phase 3 == Phase 1 (deterministic).
With MemoryBank, Phase 3 should be >= Phase 1 (memory helps).

Usage:
    cd /Users/youest/sviluppo/nanochat
    uv run python -m scripts.test3_persistent_memory --device cpu --n_steps 300
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from nanochat.cellmem import CellMem, CellMemConfig
from nanochat.memory_bank import MemoryBank, MemoryBankConfig


@dataclass
class TinyConfig:
    vocab_size: int = 32
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    max_seq_len: int = 40


class TinyBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, g_attn=None, g_mlp=None, r_add=None):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                                attn_mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device))
        if g_attn is not None:
            attn_out = attn_out * torch.sigmoid(g_attn)
        x = x + attn_out
        if r_add is not None:
            x = x + r_add
        mlp_out = self.mlp(self.ln2(x))
        if g_mlp is not None:
            mlp_out = mlp_out * torch.sigmoid(g_mlp)
        x = x + mlp_out
        return x


class TinyTransformerWithMemory(nn.Module):
    """Tiny transformer with CellMem + MemoryBank integration.

    Integration wiring (at inference, weights frozen):
    - CellMem processes tokens, produces gates + accumulates M state
    - MemoryBank.select_working_set(context) at start of sequence
    - MemoryBank.read_from_working_set(token) per token -> added to r_add
    - MemoryBank.write(key, value, surprise) at end of sequence
    """
    def __init__(self, cfg: TinyConfig, use_cellmem=False, use_bank=False, persistent_m=False):
        super().__init__()
        self.cfg = cfg
        self.use_cellmem = use_cellmem
        self.use_bank = use_bank
        self.persistent_m = persistent_m
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TinyBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

        if use_cellmem:
            cm_cfg = CellMemConfig(d_model=cfg.d_model, d_cell=16, n_cells=2)
            self.cellmem = CellMem(cm_cfg)
            self._m_initialized = False
        if use_bank:
            bank_cfg = MemoryBankConfig(
                d_key=cfg.d_model,
                d_value=16 * 2,  # d_cell * n_cells
                max_slots=256,
                working_set_k=8,
                write_threshold=0.01,
            )
            self.bank = MemoryBank(bank_cfg)
            self.mem_proj = nn.Linear(16 * 2, cfg.d_model)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)

        # MemoryBank: select working set at start of sequence
        ws_keys, ws_values = None, None
        if self.use_bank and self.bank.size > 0:
            context = x.mean(dim=(0, 1)).detach()
            ws_keys, ws_values = self.bank.select_working_set(context)

        # CellMem: process token-by-token
        if self.use_cellmem:
            if not self.persistent_m or not self._m_initialized:
                self.cellmem.reset_state(B)
                self._m_initialized = True
            all_g_attn, all_g_mlp, all_r_add = [], [], []
            for t in range(T):
                g_attn, g_mlp, r_add, x0_mod = self.cellmem(x[:, t, :])
                all_g_attn.append(g_attn)
                all_g_mlp.append(g_mlp)
                # Inject memory bank read into r_add
                if self.use_bank and ws_keys is not None and ws_keys.shape[0] > 0:
                    r_mem = self.bank.read_from_working_set(x[:, t, :].mean(dim=0), ws_keys, ws_values)
                    r_add = r_add + self.mem_proj(r_mem.unsqueeze(0).expand(B, -1))
                all_r_add.append(r_add)
            g_attn_seq = torch.stack(all_g_attn, dim=1)
            g_mlp_seq = torch.stack(all_g_mlp, dim=1)
            r_add_seq = torch.stack(all_r_add, dim=1)

        # Transformer blocks
        for block in self.blocks:
            if self.use_cellmem:
                x = block(x, g_attn=g_attn_seq, g_mlp=g_mlp_seq, r_add=r_add_seq)
            else:
                x = block(x)

        logits = self.head(x)

        # MemoryBank: write at end of sequence (only if bank enabled)
        if self.use_bank and self.use_cellmem:
            novelty = self.cellmem.get_mean_novelty()
            key = x.mean(dim=(0, 1)).detach()
            value = x[:, -1, :].mean(dim=0).detach()[:self.bank.config.d_value]
            self.bank.write(key, value, surprise=novelty)
            self.bank.decay()

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss
        return logits


# ---------------------------------------------------------------------------
# Fact generation
# ---------------------------------------------------------------------------
FACT_TOK = 26
QUERY_TOK = 27
NOISE_TOK = 28


def create_fixed_facts(n_facts=5, seed=42):
    """Create a fixed set of symbol->count mappings."""
    rng = torch.Generator().manual_seed(seed)
    symbols = torch.randperm(10, generator=rng)[:n_facts].tolist()
    counts = (torch.randint(16, 26, (n_facts,), generator=rng)).tolist()
    return {s: c for s, c in zip(symbols, counts)}


def generate_fact_batch(batch_size, facts_dict, seq_len=30, seed=None):
    """Generate sequences encoding facts from facts_dict.
    Each sequence: [FACT sym count] * n_facts + noise + [QUERY sym].
    Target: the count token at position -1."""
    if seed is not None:
        torch.manual_seed(seed)
    facts_list = list(facts_dict.items())
    n_facts = len(facts_list)
    inputs = torch.full((batch_size, seq_len), NOISE_TOK, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    for b in range(batch_size):
        pos = 0
        for sym, count in facts_list:
            if pos + 3 <= seq_len - 3:
                inputs[b, pos] = FACT_TOK
                inputs[b, pos + 1] = sym
                inputs[b, pos + 2] = count
                pos += 3
        for p in range(pos, seq_len - 3):
            inputs[b, p] = NOISE_TOK
        q_idx = torch.randint(0, n_facts, (1,)).item()
        q_sym, q_count = facts_list[q_idx]
        inputs[b, -3] = QUERY_TOK
        inputs[b, -2] = q_sym
        targets[b, -1] = q_count
    return inputs, targets


def generate_noise_batch(batch_size, vocab_size=32, seq_len=30, seed=None):
    """Generate random noise sequences (no facts, no query)."""
    if seed is not None:
        torch.manual_seed(seed)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    return inputs


def assess_on_facts(model, facts_dict, n_batches=5, batch_size=64, device='cpu'):
    """Assess model accuracy on facts (no training, no gradient)."""
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(n_batches):
            inputs, targets = generate_fact_batch(batch_size, facts_dict, seed=77777 + i)
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits[:, -1, :].argmax(dim=-1)
            mask = targets[:, -1] != -1
            if mask.any():
                correct += (preds[mask] == targets[:, -1][mask]).sum().item()
                total += mask.sum().item()
    return correct / max(total, 1) * 100


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def generate_query_only_batch(batch_size, facts_dict, seq_len=30, seed=None):
    """Generate sequences with ONLY a query (no facts in context).
    The model must recall from memory, not from the current sequence."""
    if seed is not None:
        torch.manual_seed(seed)
    facts_list = list(facts_dict.items())
    n_facts = len(facts_list)
    inputs = torch.full((batch_size, seq_len), NOISE_TOK, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    for b in range(batch_size):
        q_idx = torch.randint(0, n_facts, (1,)).item()
        q_sym, q_count = facts_list[q_idx]
        inputs[b, -3] = QUERY_TOK
        inputs[b, -2] = q_sym
        targets[b, -1] = q_count
    return inputs, targets


def assess_cross_sequence(model, facts_dict, n_batches=5, batch_size=64, device='cpu'):
    """Assess: can the model answer queries when facts are NOT in the current sequence?
    This is the real test for cross-sequence memory."""
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(n_batches):
            inputs, targets = generate_query_only_batch(batch_size, facts_dict, seed=88888 + i)
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits[:, -1, :].argmax(dim=-1)
            mask = targets[:, -1] != -1
            if mask.any():
                correct += (preds[mask] == targets[:, -1][mask]).sum().item()
                total += mask.sum().item()
    return correct / max(total, 1) * 100


def generate_random_facts(n_facts=5, seed=None):
    """Generate random facts (different every call if seed changes)."""
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    symbols = torch.randperm(10, generator=rng)[:n_facts].tolist()
    counts = (torch.randint(16, 26, (n_facts,), generator=rng)).tolist()
    return {s: c for s, c in zip(symbols, counts)}


def run_experiment(variant, n_train_steps, n_inference_steps, device, seed, batch_size=64):
    """Run the inference-persistence experiment with NOVEL facts.

    Phase 0: Train on RANDOM facts each batch (model learns task STRUCTURE,
             not specific facts). Then freeze weights.
    Phase 1: Show NOVEL facts (never seen in training) in context.
             MemoryBank accumulates. Test cross-sequence recall.
    Phase 2: Show noise sequences (interference on M/bank).
    Phase 3: Query the novel facts without context (cross-sequence recall).

    Without MemoryBank: ~random chance on novel facts cross-sequence.
    With MemoryBank: stored embeddings should help recall.
    """
    torch.manual_seed(seed)
    cfg = TinyConfig()

    use_cm = variant in ("cellmem_reset", "cellmem_persistent")
    use_bank = variant == "cellmem_persistent"
    persistent_m = variant == "cellmem_persistent"

    # During training, always use non-persistent M (avoids graph issues)
    model = TinyTransformerWithMemory(
        cfg, use_cellmem=use_cm, use_bank=use_bank, persistent_m=False
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === Phase 0: TRAIN on random facts (learn task structure, not specific facts) ===
    model.train()
    for step in range(n_train_steps):
        # Different random facts each batch — model can't memorize specifics
        train_facts = generate_random_facts(n_facts=5, seed=seed * 10000 + step)
        inputs, targets = generate_fact_batch(batch_size, train_facts, seed=seed * 10000 + step + 1)
        inputs, targets = inputs.to(device), targets.to(device)
        loss = model(inputs, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Test on in-context facts to verify model learned the task structure
    test_facts = create_fixed_facts(n_facts=5, seed=seed + 999)
    train_acc = assess_on_facts(model, test_facts, device=device)

    # === FREEZE all weights ===
    model.requires_grad_(False)

    # Enable persistent M for inference
    model.persistent_m = persistent_m
    if use_cm:
        model.cellmem.reset_state(batch_size)
        model._m_initialized = False
    if use_bank:
        model.bank = MemoryBank(model.bank.config)

    # NOVEL facts: never seen during training
    novel_facts = create_fixed_facts(n_facts=5, seed=seed + 5000)

    results = {"train_in_context": train_acc}

    # Before: query novel facts cross-sequence (should be ~random)
    results["before"] = assess_cross_sequence(model, novel_facts, device=device)

    # === Phase 1: LEARN — show novel facts in context (bank accumulates) ===
    with torch.no_grad():
        for step in range(n_inference_steps):
            inputs, targets = generate_fact_batch(batch_size, novel_facts, seed=seed * 20000 + step)
            inputs, targets = inputs.to(device), targets.to(device)
            model(inputs)  # bank writes, M accumulates
    results["after_learn"] = assess_cross_sequence(model, novel_facts, device=device)
    bank_size_after_learn = model.bank.size if use_bank else 0

    # === Phase 2: INTERFERE (noise) ===
    with torch.no_grad():
        for step in range(n_inference_steps):
            inputs = generate_noise_batch(batch_size, seed=seed * 30000 + step)
            inputs = inputs.to(device)
            model(inputs)
    results["after_noise"] = assess_cross_sequence(model, novel_facts, device=device)
    bank_size_after_interfere = model.bank.size if use_bank else 0

    # === Phase 3: RECALL ===
    results["recall"] = assess_cross_sequence(model, novel_facts, device=device)

    return results, bank_size_after_learn, bank_size_after_interfere


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 3: Persistent Memory (Inference-Only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_steps", type=int, default=300, help="Training steps in Phase 0")
    parser.add_argument("--n_inference", type=int, default=50, help="Inference steps per phase")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    print(f"Config: n_train={args.n_steps}, n_inference={args.n_inference}, "
          f"batch_size={args.batch_size}, device={args.device}")
    print(f"Seeds: {args.seeds}")
    print()
    print("Phase 0: TRAIN on RANDOM facts each batch (learn task structure, not specific facts)")
    print("Phase 1: LEARN - show NOVEL facts in context (MemoryBank accumulates)")
    print("Phase 2: INTERFERE - inference with noise")
    print("Phase 3: RECALL - query novel facts WITHOUT them in context (cross-sequence)")
    print()
    print("All accuracy columns are CROSS-SEQUENCE: facts NOT in query context")
    print(f"Random chance: ~{100/10:.0f}% (10 possible count tokens)")
    print()

    all_results = {}
    for variant in ["baseline", "cellmem_reset", "cellmem_persistent"]:
        print(f"{'='*70}")
        print(f"Variant: {variant}")
        variant_results = []
        for seed in args.seeds:
            results, bank_learn, bank_interfere = run_experiment(
                variant, args.n_steps, args.n_inference, args.device, seed, args.batch_size
            )
            variant_results.append(results)
            bank_info = f"  bank: {bank_learn}/{bank_interfere} slots" if variant == "cellmem_persistent" else ""
            print(f"  Seed {seed}: in_ctx={results['train_in_context']:.1f}%  "
                  f"before={results['before']:.1f}%  after_learn={results['after_learn']:.1f}%  "
                  f"after_noise={results['after_noise']:.1f}%  recall={results['recall']:.1f}%{bank_info}")
        mean = {k: sum(r[k] for r in variant_results) / len(variant_results) for k in variant_results[0]}
        print(f"  MEAN:    in_ctx={mean['train_in_context']:.1f}%  before={mean['before']:.1f}%  "
              f"after_learn={mean['after_learn']:.1f}%  after_noise={mean['after_noise']:.1f}%  "
              f"recall={mean['recall']:.1f}%")
        all_results[variant] = mean

    print(f"\n{'='*70}")
    print("SUMMARY (cross-sequence recall of NOVEL facts)")
    print(f"{'Variant':<25} {'InCtx':>8} {'Before':>8} {'After Learn':>12} {'After Noise':>12} {'Recall':>8}")
    print("-" * 70)
    for variant in ["baseline", "cellmem_reset", "cellmem_persistent"]:
        r = all_results[variant]
        print(f"{variant:<25} {r['train_in_context']:>7.1f}% {r['before']:>7.1f}% "
              f"{r['after_learn']:>11.1f}% {r['after_noise']:>11.1f}% {r['recall']:>7.1f}%")
