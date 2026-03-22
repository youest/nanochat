"""
Test 2b: Accumulation Tasks — testing CellMem on tasks where ATTENTION ALONE IS NOT ENOUGH.

Two tasks designed to expose the limits of pure attention:

Task 1: Counting
  Sequence: BOS A B C A D A B A QUERY A COUNT
  The model must count how many times a queried symbol appeared.
  Attention can FIND all positions where X appears, but can't COUNT them —
  there's no natural "counting" operation in attention.
  CellMem should help: M can accumulate a representation that grows with occurrences.

Task 2: Pattern Shift Detection
  Sequence: BOS A=1 B=2 ... SHIFT A=3 B=1 ... QUERY A TARGET 3
  First half: consistent key->value mapping. After SHIFT: mapping CHANGES.
  The model must use the NEW mapping, ignoring the old one.
  Attention sees ALL tokens equally — no built-in notion of "recency after SHIFT."
  CellMem should help: anti-Hebbian novelty-driven update causes M to adapt at SHIFT.

Compares three variants:
  - baseline: plain tiny transformer
  - cellmem: tiny transformer + CellMem module in each block
  - ttt: tiny transformer + TTT layer in each block (same interface as CellMem)

Usage:
    cd /Users/youest/sviluppo/nanochat
    uv run python -m scripts.test2b_accumulation_tasks --device cpu --n_steps 3000
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.cellmem import CellMem, CellMemConfig

# TTT is being implemented by another agent — gracefully degrade if unavailable
try:
    from nanochat.ttt import TTTLayer, TTTConfig
    HAS_TTT = True
except ImportError:
    HAS_TTT = False
    print("[WARN] nanochat.ttt not available, TTT variant will be skipped.")


# ---------------------------------------------------------------------------
# 1. Task Generators
# ---------------------------------------------------------------------------

def generate_counting_batch(batch_size, seq_len=30, n_symbols=16, max_count=15, device='cpu'):
    """
    Generate a batch of counting task sequences.

    Sequence format:
        BOS sym0 sym1 ... sym_{seq_len-1} QUERY query_sym
    Target: count_start + count(query_sym in sequence), capped at max_count.

    Token layout:
        0: BOS, 1: QUERY  (special)
        2..2+n_symbols-1: symbol tokens
        2+n_symbols..2+n_symbols+max_count: count tokens

    Returns:
        input_ids: (B, 1 + seq_len + 2) = (B, total_len)
        targets: (B,) count of queried symbol (as token ids in count range)
        vocab_size: int
    """
    BOS, QUERY = 0, 1
    N_SPECIAL = 2
    symbol_start = N_SPECIAL                       # symbols: 2..17
    count_start = N_SPECIAL + n_symbols            # counts: 18..33
    vocab_size = N_SPECIAL + n_symbols + max_count + 1  # 34

    total_len = 1 + seq_len + 2
    input_ids = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Random sequence of symbols
        symbols = torch.randint(0, n_symbols, (seq_len,))
        input_ids[b, 0] = BOS
        input_ids[b, 1:1 + seq_len] = symbols + symbol_start

        # Pick a query symbol
        query_sym = torch.randint(0, n_symbols, (1,)).item()
        input_ids[b, 1 + seq_len] = QUERY
        input_ids[b, 1 + seq_len + 1] = query_sym + symbol_start

        # Count occurrences
        count = (symbols == query_sym).sum().item()
        count = min(count, max_count)
        targets[b] = count_start + count

    return input_ids, targets, vocab_size


def generate_pattern_shift_batch(batch_size, n_keys=8, n_values=8, n_examples_per_half=4, device='cpu'):
    """
    Generate a batch of pattern shift detection sequences.

    Sequence format:
        BOS [K EQ V SEP]*n_examples_per_half SHIFT [K EQ V SEP]*n_examples_per_half QUERY Kq EQ
    Target: the post-shift value for the queried key.

    Token layout:
        0: BOS, 1: EQ, 2: SEP, 3: SHIFT, 4: QUERY  (special)
        5..5+n_keys-1: key tokens
        5+n_keys..5+n_keys+n_values-1: value tokens

    Returns:
        input_ids: (B, seq_len)
        targets: (B,) the correct post-shift value for the queried key
        vocab_size: int
    """
    BOS, EQ, SEP, SHIFT, QUERY = 0, 1, 2, 3, 4
    N_SPECIAL = 5
    key_start = N_SPECIAL                   # keys: 5..12
    val_start = N_SPECIAL + n_keys          # values: 13..20
    vocab_size = N_SPECIAL + n_keys + n_values  # 21

    # Sequence structure:
    # BOS + pre_shift_pairs(K EQ V SEP each = 4*n) + SHIFT + post_shift_pairs(4*n) + QUERY Kq EQ
    seq_len = 1 + n_examples_per_half * 4 + 1 + n_examples_per_half * 4 + 3

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Create two DIFFERENT mappings
        keys = torch.randperm(n_keys)[:n_examples_per_half]

        perm1 = torch.randperm(n_values)[:n_examples_per_half]  # pre-shift mapping
        perm2 = torch.randperm(n_values)[:n_examples_per_half]  # post-shift mapping
        # Ensure they're different for at least one key
        while (perm1 == perm2).all():
            perm2 = torch.randperm(n_values)[:n_examples_per_half]

        pos = 0
        input_ids[b, pos] = BOS; pos += 1

        # Pre-shift examples
        for i in range(n_examples_per_half):
            input_ids[b, pos] = keys[i] + key_start; pos += 1
            input_ids[b, pos] = EQ; pos += 1
            input_ids[b, pos] = perm1[i] + val_start; pos += 1
            input_ids[b, pos] = SEP; pos += 1

        input_ids[b, pos] = SHIFT; pos += 1

        # Post-shift examples
        for i in range(n_examples_per_half):
            input_ids[b, pos] = keys[i] + key_start; pos += 1
            input_ids[b, pos] = EQ; pos += 1
            input_ids[b, pos] = perm2[i] + val_start; pos += 1
            input_ids[b, pos] = SEP; pos += 1

        # Query
        query_idx = torch.randint(0, n_examples_per_half, (1,)).item()
        input_ids[b, pos] = QUERY; pos += 1
        input_ids[b, pos] = keys[query_idx] + key_start; pos += 1
        input_ids[b, pos] = EQ; pos += 1

        targets[b] = perm2[query_idx] + val_start  # POST-shift value

    return input_ids, targets, vocab_size


# ---------------------------------------------------------------------------
# 2. Tiny Transformer (copied from test2_associative_recall.py for independence)
# ---------------------------------------------------------------------------

@dataclass
class TinyTransformerConfig:
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 64
    vocab_size: int = 46
    seq_len: int = 64
    dropout: float = 0.0
    use_cellmem: bool = False
    use_ttt: bool = False
    cellmem_config: Optional[CellMemConfig] = None
    ttt_config: object = None  # TTTConfig when available


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class TinyMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class TinyBlock(nn.Module):
    def __init__(self, config, cellmem_module=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config.d_model, config.n_head)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = TinyMLP(config.d_model)
        self.cellmem = cellmem_module  # CellMem, TTTLayer, or None

    def forward(self, x, cellmem_outputs=None):
        """
        x: (B, T, d_model)
        cellmem_outputs: tuple of (g_attn, g_mlp, r_add, x0_mod) each (B, T, d_model), or None
        """
        attn_out = self.attn(self.ln1(x))
        if cellmem_outputs is not None:
            g_attn, g_mlp, r_add, _ = cellmem_outputs
            attn_out = attn_out * torch.sigmoid(g_attn)
            x = x + attn_out + r_add
        else:
            x = x + attn_out

        mlp_out = self.mlp(self.ln2(x))
        if cellmem_outputs is not None:
            _, g_mlp, _, _ = cellmem_outputs
            mlp_out = mlp_out * torch.sigmoid(g_mlp)
        x = x + mlp_out
        return x


def _move_mem_state_to_device(mem, device):
    """Move CellMem/TTT internal state tensors to the correct device after reset_state."""
    if isinstance(mem, CellMem):
        mem._M = [m.to(device) for m in mem._M]
        mem._T = [t.to(device) for t in mem._T]
        mem._astro_mu = [mu.to(device) for mu in mem._astro_mu]
        mem._astro_sigma = [sigma.to(device) for sigma in mem._astro_sigma]
        mem._novelty = [n.to(device) for n in mem._novelty]
    elif HAS_TTT and isinstance(mem, TTTLayer):
        mem._W_inner = [w.to(device) for w in mem._W_inner]


class TinyTransformer(nn.Module):
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.seq_len, config.d_model)

        # Build blocks, each optionally with a CellMem/TTT module
        self.blocks = nn.ModuleList()
        self.mem_modules = nn.ModuleList()  # parallel list of CellMem/TTT (or empty placeholder)

        for _ in range(config.n_layer):
            if config.use_cellmem and config.cellmem_config is not None:
                mem = CellMem(config.cellmem_config)
            elif config.use_ttt and config.ttt_config is not None and HAS_TTT:
                mem = TTTLayer(config.ttt_config)
            else:
                mem = None

            block = TinyBlock(config, cellmem_module=mem)
            self.blocks.append(block)
            # Store mem modules separately so they are registered with nn.Module
            if mem is not None:
                self.mem_modules.append(mem)

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device

        tok_emb = self.embedding(input_ids)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            mem = block.cellmem
            if mem is not None:
                # CellMem/TTT processes tokens sequentially: reset state, loop over T
                mem.reset_state(B)
                _move_mem_state_to_device(mem, device)
                g_attn_list, g_mlp_list, r_add_list, x0_mod_list = [], [], [], []
                for t in range(T):
                    x_t = x[:, t, :]  # (B, d_model)
                    g_attn_t, g_mlp_t, r_add_t, x0_mod_t = mem(x_t)
                    g_attn_list.append(g_attn_t)
                    g_mlp_list.append(g_mlp_t)
                    r_add_list.append(r_add_t)
                    x0_mod_list.append(x0_mod_t)
                # Stack into (B, T, d_model) tensors
                cellmem_outputs = (
                    torch.stack(g_attn_list, dim=1),
                    torch.stack(g_mlp_list, dim=1),
                    torch.stack(r_add_list, dim=1),
                    torch.stack(x0_mod_list, dim=1),
                )
                x = block(x, cellmem_outputs=cellmem_outputs)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------

def train_and_evaluate(model_name, model, generate_fn, n_steps=3000, batch_size=64,
                       lr=3e-4, device='cpu', seed=42):
    """
    Train the model using a task-specific batch generator.

    Args:
        model_name: label for logging
        model: TinyTransformer
        generate_fn: callable(batch_size, device) -> (input_ids, targets, vocab_size)
        n_steps: training steps
        batch_size: batch size
        lr: learning rate
        device: device string
        seed: random seed
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    metrics = {
        'loss_history': [],
        'acc_history': [],
    }

    for step in range(n_steps):
        input_ids, targets, _ = generate_fn(batch_size, device=device)

        logits = model(input_ids)           # (B, T, vocab_size)
        pred_logits = logits[:, -1, :]      # (B, vocab_size) — last position = answer
        loss = F.cross_entropy(pred_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (pred_logits.argmax(dim=-1) == targets).float().mean().item()
        metrics['loss_history'].append(loss.item())
        metrics['acc_history'].append(acc)

        if step % 200 == 0:
            window = min(50, len(metrics['acc_history']))
            recent_acc = sum(metrics['acc_history'][-window:]) / window
            recent_loss = sum(metrics['loss_history'][-window:]) / window
            print(f"[{model_name}] Step {step}: loss={recent_loss:.4f} acc={recent_acc:.2%}")

    return metrics


# ---------------------------------------------------------------------------
# 4. Model Factory
# ---------------------------------------------------------------------------

def make_model(variant, vocab_size, seq_len, d_model=64, n_layer=2, n_head=2,
               d_cell=16, n_cells=2):
    """Create a TinyTransformer for the given variant and task parameters."""
    config = TinyTransformerConfig(
        n_layer=n_layer,
        n_head=n_head,
        d_model=d_model,
        vocab_size=vocab_size,
        seq_len=seq_len,
        use_cellmem=False,
        use_ttt=False,
    )

    if variant == 'cellmem':
        config.use_cellmem = True
        config.cellmem_config = CellMemConfig(d_model=d_model, d_cell=d_cell, n_cells=n_cells)
    elif variant == 'ttt':
        if not HAS_TTT:
            return None
        config.use_ttt = True
        config.ttt_config = TTTConfig(d_model=d_model, d_cell=d_cell, n_cells=n_cells)

    model = TinyTransformer(config)
    return model


# ---------------------------------------------------------------------------
# 5. Task Definitions
# ---------------------------------------------------------------------------

TASK_DEFS = {
    'counting': {
        'description': 'Count occurrences of a queried symbol in a random sequence',
        'random_chance': '1/16 = 6.25%',
    },
    'pattern_shift': {
        'description': 'Recall post-shift key-value mapping, ignoring pre-shift mapping',
        'random_chance': '1/8 = 12.50%',
    },
}


def get_task_params(task_name):
    """Return (generate_fn, vocab_size, seq_len) for the given task."""
    if task_name == 'counting':
        seq_len_counting = 30
        n_symbols = 16
        max_count = 15

        def gen_fn(batch_size, device='cpu'):
            return generate_counting_batch(
                batch_size, seq_len=seq_len_counting, n_symbols=n_symbols,
                max_count=max_count, device=device,
            )

        # Probe vocab_size from a dummy batch
        _, _, vocab_size = generate_counting_batch(1, seq_len=seq_len_counting,
                                                    n_symbols=n_symbols, max_count=max_count)
        total_seq_len = 1 + seq_len_counting + 2  # BOS + symbols + QUERY + query_sym = 33
        return gen_fn, vocab_size, total_seq_len

    elif task_name == 'pattern_shift':
        n_keys = 8
        n_values = 8
        n_examples_per_half = 4

        def gen_fn(batch_size, device='cpu'):
            return generate_pattern_shift_batch(
                batch_size, n_keys=n_keys, n_values=n_values,
                n_examples_per_half=n_examples_per_half, device=device,
            )

        _, _, vocab_size = generate_pattern_shift_batch(1, n_keys=n_keys, n_values=n_values,
                                                         n_examples_per_half=n_examples_per_half)
        # seq_len = 1 + 4*n_examples_per_half + 1 + 4*n_examples_per_half + 3 = 1+16+1+16+3 = 37
        total_seq_len = 1 + n_examples_per_half * 4 + 1 + n_examples_per_half * 4 + 3
        return gen_fn, vocab_size, total_seq_len

    else:
        raise ValueError(f"Unknown task: {task_name}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test 2b: Accumulation Tasks — CellMem on tasks where attention alone is not enough"
    )
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'],
                        help='Device to use (default: cpu)')
    parser.add_argument('--n_steps', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--task', choices=['counting', 'pattern_shift', 'both'], default='both')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    task_names = ['counting', 'pattern_shift'] if args.task == 'both' else [args.task]

    for task_name in task_names:
        print(f"\n{'#' * 60}")
        print(f"# TASK: {task_name}")
        print(f"# {TASK_DEFS[task_name]['description']}")
        print(f"# Random chance: {TASK_DEFS[task_name]['random_chance']}")
        print(f"{'#' * 60}")

        generate_fn, vocab_size, seq_len = get_task_params(task_name)
        print(f"  vocab_size={vocab_size}, seq_len={seq_len}")

        # --- Print parameter counts ---
        print("\n" + "=" * 60)
        print("PARAMETER COUNTS")
        print("=" * 60)
        variants = ['baseline', 'cellmem']
        if HAS_TTT:
            variants.append('ttt')
        param_counts = {}
        for variant in variants:
            m = make_model(variant, vocab_size=vocab_size, seq_len=seq_len)
            if m is None:
                continue
            n = count_params(m)
            param_counts[variant] = n
            print(f"  {variant:>10s}: {n:,} parameters")
        if 'cellmem' in param_counts and 'baseline' in param_counts:
            overhead = param_counts['cellmem'] - param_counts['baseline']
            pct = 100 * overhead / param_counts['baseline']
            print(f"  CellMem overhead: +{overhead:,} params ({pct:.1f}%)")
        if 'ttt' in param_counts and 'baseline' in param_counts:
            overhead = param_counts['ttt'] - param_counts['baseline']
            pct = 100 * overhead / param_counts['baseline']
            print(f"  TTT overhead:     +{overhead:,} params ({pct:.1f}%)")

        # --- Train all variants ---
        results = {}
        for seed in args.seeds:
            for variant in variants:
                model = make_model(variant, vocab_size=vocab_size, seq_len=seq_len)
                if model is None:
                    print(f"\n[SKIP] {variant} (not available)")
                    continue
                model = model.to(device)
                n_params = count_params(model)
                print(f"\n{'=' * 60}")
                print(f"Variant: {variant} | Seed: {seed} | Params: {n_params:,}")
                print(f"{'=' * 60}")

                metrics = train_and_evaluate(
                    model_name=f"{variant}/seed{seed}",
                    model=model,
                    generate_fn=generate_fn,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    seed=seed,
                )
                results[f"{variant}/seed{seed}"] = metrics
                # Free memory
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()

        # --- Summary for this task ---
        print(f"\n{'=' * 60}")
        print(f"SUMMARY — {task_name}")
        print(f"Random chance: {TASK_DEFS[task_name]['random_chance']}")
        print(f"{'=' * 60}")
        for variant in variants:
            accs = []
            for seed in args.seeds:
                key = f"{variant}/seed{seed}"
                if key not in results:
                    continue
                window = min(100, len(results[key]['acc_history']))
                final_acc = sum(results[key]['acc_history'][-window:]) / window
                accs.append(final_acc)
            if not accs:
                print(f"  {variant:>10s}: SKIPPED")
                continue
            mean_acc = sum(accs) / len(accs)
            std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
            print(f"  {variant:>10s}: {mean_acc:.1%} +/- {std_acc:.1%}  (seeds: {[f'{a:.1%}' for a in accs]})")

    print("\nDone.")


if __name__ == '__main__':
    main()
