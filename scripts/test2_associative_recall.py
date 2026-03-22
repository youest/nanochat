"""
Test 2: Associative Recall — does CellMem improve in-context learning?

Synthetic task: given key-value pairs in context, predict the value for a queried key.
Each batch has NEW random associations, so the model CANNOT memorize — it MUST read context.

Compares three variants:
  - baseline: plain tiny transformer
  - cellmem: tiny transformer + CellMem module in each block
  - ttt: tiny transformer + TTT layer in each block (same interface as CellMem)

Usage:
    cd /Users/youest/sviluppo/nanochat
    uv run python scripts/test2_associative_recall.py --device mps --n_steps 2000
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
# 1. Associative Recall Dataset
# ---------------------------------------------------------------------------

# Special tokens
BOS = 0
EQ = 1
SEP = 2
QUERY = 3
N_SPECIAL = 4

def generate_batch(batch_size, n_pairs, n_keys=26, n_values=16, device='cpu'):
    """
    Generate a batch of associative recall sequences.

    Sequence format:
        BOS K0 EQ V0 SEP K1 EQ V1 SEP ... QUERY Kq EQ
    Target: the value token paired with the queried key Kq.

    Returns:
        input_ids: (B, seq_len) token sequence
        targets: (B,) the correct value token for the query
        seq_len: int, length of each sequence
    """
    # Token ranges
    key_start = N_SPECIAL
    val_start = N_SPECIAL + n_keys

    # seq_len = 1 (BOS) + n_pairs * 3 (K EQ V) + (n_pairs - 1) (SEP between pairs) + 2 (QUERY K EQ)
    # Actually: BOS K0 EQ V0 SEP K1 EQ V1 SEP ... SEP K_{n-1} EQ V_{n-1} QUERY Kq EQ
    # = 1 + n_pairs*3 + (n_pairs-1) + 3
    # = 1 + 3*n_pairs + n_pairs - 1 + 3 = 4*n_pairs + 3
    seq_len = 4 * n_pairs + 3

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Sample n_pairs unique keys (without replacement)
        key_indices = torch.randperm(n_keys)[:n_pairs]
        keys = key_indices + key_start
        # Sample n_pairs values (with replacement)
        vals = torch.randint(0, n_values, (n_pairs,)) + val_start

        # Build the sequence: BOS K0 EQ V0 SEP K1 EQ V1 SEP ... QUERY Kq EQ
        pos = 0
        input_ids[b, pos] = BOS; pos += 1
        for p in range(n_pairs):
            input_ids[b, pos] = keys[p]; pos += 1
            input_ids[b, pos] = EQ; pos += 1
            input_ids[b, pos] = vals[p]; pos += 1
            if p < n_pairs - 1:
                input_ids[b, pos] = SEP; pos += 1
        # Query section
        query_idx = torch.randint(0, n_pairs, (1,)).item()
        input_ids[b, pos] = QUERY; pos += 1
        input_ids[b, pos] = keys[query_idx]; pos += 1
        input_ids[b, pos] = EQ; pos += 1
        assert pos == seq_len

        targets[b] = vals[query_idx]

    return input_ids, targets, seq_len

# ---------------------------------------------------------------------------
# 2. Tiny Transformer (self-contained, no nanochat.gpt dependency)
# ---------------------------------------------------------------------------

@dataclass
class TinyTransformerConfig:
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 64
    vocab_size: int = 46   # 4 + 26 + 16
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

# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------

def train_and_evaluate(model_name, model, n_steps=2000, batch_size=32, n_pairs=8,
                       n_keys=26, n_values=16, lr=3e-4, device='cpu', seed=42):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    metrics = {
        'loss_history': [],
        'acc_history': [],
    }

    for step in range(n_steps):
        input_ids, targets, seq_len = generate_batch(batch_size, n_pairs,
                                                      n_keys=n_keys, n_values=n_values,
                                                      device=device)

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
# 4. Ablation helpers
# ---------------------------------------------------------------------------

class CellMemNoTopology(CellMem):
    """CellMem with topology T frozen to ones (no update)."""
    def forward(self, x):
        # Freeze T by resetting it every call
        for i in range(self.config.n_cells):
            self._T[i] = torch.ones_like(self._T[i])
        return super().forward(x)


class CellMemNoAstrocyte(CellMem):
    """CellMem without astrocyte modulation (alpha_eff = alpha_base, no sigma/mu scaling)."""
    def forward(self, x):
        B, d_model = x.shape
        n_cells = self.config.n_cells
        d_slice = d_model // n_cells

        out_attn, out_mlp, out_add, out_x0 = [], [], [], []

        for i in range(n_cells):
            x_slice = x[:, i * d_slice:(i + 1) * d_slice]
            z = x_slice @ self.W_in[i]
            M, T = self._M[i], self._T[i]
            z_pred = (M * T) @ z.T
            z_pred = z_pred.T
            error = z - z_pred
            # Use alpha_base directly (no astrocyte modulation)
            alpha_eff = self.alpha_base[i]
            # Anti-Hebbian update
            error_d = error.detach()
            z_d = z.detach()
            delta_M = alpha_eff.detach() * (error_d.unsqueeze(-1) * z_d.unsqueeze(-2)).mean(dim=0)
            self._M[i] = M + delta_M
            # Topology update (unchanged)
            gamma_val = self.gamma[i].detach()
            tau_val = self.tau[i].detach()
            delta_T = gamma_val * (
                (error_d.abs().unsqueeze(-1) * z_d.abs().unsqueeze(-2)).mean(dim=0) - tau_val * T
            )
            self._T[i] = (T + delta_T).clamp(0, 1)
            # MSB fan-out
            mem_out = (self._M[i] * self._T[i]) @ z.T
            mem_out = mem_out.T
            out_attn.append(mem_out @ self.W_msb[i][0])
            out_mlp.append(mem_out @ self.W_msb[i][1])
            out_add.append(mem_out @ self.W_msb[i][2])
            out_x0.append(mem_out @ self.W_msb[i][3])

        return (torch.cat(out_attn, dim=-1), torch.cat(out_mlp, dim=-1),
                torch.cat(out_add, dim=-1), torch.cat(out_x0, dim=-1))


class CellMemHebbian(CellMem):
    """CellMem with Hebbian update: outer(z, z) instead of outer(error, z)."""
    def forward(self, x):
        B, d_model = x.shape
        n_cells = self.config.n_cells
        d_slice = d_model // n_cells

        out_attn, out_mlp, out_add, out_x0 = [], [], [], []

        for i in range(n_cells):
            x_slice = x[:, i * d_slice:(i + 1) * d_slice]
            z = x_slice @ self.W_in[i]
            M, T = self._M[i], self._T[i]
            z_pred = (M * T) @ z.T
            z_pred = z_pred.T
            error = z - z_pred
            z_norm_sq = z.norm(dim=-1, keepdim=True) ** 2 + 1e-8
            novelty = (error.norm(dim=-1, keepdim=True) ** 2) / z_norm_sq
            self._novelty[i] = novelty.mean().detach()
            # Astrocyte modulation (same as original)
            z_norms = z.norm(dim=-1)
            mu = self._astro_mu[i]
            sigma = self._astro_sigma[i]
            mu = 0.99 * mu + 0.01 * z_norms.mean().detach()
            sigma = 0.99 * sigma + 0.01 * ((z_norms.detach() - mu) ** 2).mean()
            self._astro_mu[i] = mu
            self._astro_sigma[i] = sigma
            alpha_eff = self.alpha_base[i] * (sigma / (mu + 1e-8))
            # Hebbian update: outer(z, z) instead of outer(error, z)
            z_d = z.detach()
            delta_M = alpha_eff.detach() * (z_d.unsqueeze(-1) * z_d.unsqueeze(-2)).mean(dim=0)
            self._M[i] = M + delta_M
            # Topology update
            error_d = error.detach()
            gamma_val = self.gamma[i].detach()
            tau_val = self.tau[i].detach()
            delta_T = gamma_val * (
                (error_d.abs().unsqueeze(-1) * z_d.abs().unsqueeze(-2)).mean(dim=0) - tau_val * T
            )
            self._T[i] = (T + delta_T).clamp(0, 1)
            # MSB fan-out
            mem_out = (self._M[i] * self._T[i]) @ z.T
            mem_out = mem_out.T
            out_attn.append(mem_out @ self.W_msb[i][0])
            out_mlp.append(mem_out @ self.W_msb[i][1])
            out_add.append(mem_out @ self.W_msb[i][2])
            out_x0.append(mem_out @ self.W_msb[i][3])

        return (torch.cat(out_attn, dim=-1), torch.cat(out_mlp, dim=-1),
                torch.cat(out_add, dim=-1), torch.cat(out_x0, dim=-1))


class CellMemSingleMSB(CellMem):
    """CellMem with only 1 MSB output (r_add), others are zeros."""
    def forward(self, x):
        g_attn, g_mlp, r_add, x0_mod = super().forward(x)
        zeros = torch.zeros_like(g_attn)
        return zeros, zeros, r_add, zeros


# ---------------------------------------------------------------------------
# 5. Model factory
# ---------------------------------------------------------------------------

def make_model(variant, d_model=64, n_layer=2, n_head=2, n_keys=26, n_values=16,
               n_pairs=8, d_cell=16, n_cells=2, ablation=None):
    """Create a TinyTransformer for the given variant."""
    vocab_size = N_SPECIAL + n_keys + n_values
    seq_len = 4 * n_pairs + 3

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
    elif variant.startswith('ablation_'):
        # Ablation variants still use the cellmem slot
        config.use_cellmem = True
        config.cellmem_config = CellMemConfig(d_model=d_model, d_cell=d_cell, n_cells=n_cells)

    model = TinyTransformer(config)

    # For ablation variants, replace the CellMem modules with the ablation class
    if variant.startswith('ablation_'):
        ablation_cls_map = {
            'ablation_no_topology': CellMemNoTopology,
            'ablation_no_astrocyte': CellMemNoAstrocyte,
            'ablation_hebbian': CellMemHebbian,
            'ablation_single_msb': CellMemSingleMSB,
        }
        cls = ablation_cls_map[variant]
        cm_config = CellMemConfig(d_model=d_model, d_cell=d_cell, n_cells=n_cells)
        for block in model.blocks:
            if block.cellmem is not None:
                new_mem = cls(cm_config)
                block.cellmem = new_mem
        # Re-register mem_modules
        model.mem_modules = nn.ModuleList([b.cellmem for b in model.blocks if b.cellmem is not None])

    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test 2: Associative Recall benchmark for CellMem")
    parser.add_argument('--device', default=None, choices=['cpu', 'mps', 'cuda'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--n_pairs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_keys', type=int, default=26)
    parser.add_argument('--n_values', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--d_cell', type=int, default=16)
    parser.add_argument('--n_cells', type=int, default=2)
    parser.add_argument('--skip_ablations', action='store_true', help='Skip ablation study')
    parser.add_argument('--skip_scaling', action='store_true', help='Skip scaling test')
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # --- Print parameter counts for all variants ---
    print("\n" + "=" * 60)
    print("PARAMETER COUNTS")
    print("=" * 60)
    variants_main = ['baseline', 'cellmem']
    if HAS_TTT:
        variants_main.append('ttt')
    param_counts = {}
    for variant in variants_main:
        m = make_model(variant, d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
                       n_keys=args.n_keys, n_values=args.n_values, n_pairs=args.n_pairs,
                       d_cell=args.d_cell, n_cells=args.n_cells)
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

    # --- Main comparison ---
    results = {}
    for seed in args.seeds:
        for variant in variants_main:
            model = make_model(variant, d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
                               n_keys=args.n_keys, n_values=args.n_values, n_pairs=args.n_pairs,
                               d_cell=args.d_cell, n_cells=args.n_cells)
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
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_pairs=args.n_pairs,
                n_keys=args.n_keys,
                n_values=args.n_values,
                lr=args.lr,
                device=device,
                seed=seed,
            )
            results[f"{variant}/seed{seed}"] = metrics
            # Free memory
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY — Main Comparison")
    print("=" * 60)
    for variant in variants_main:
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

    # --- Ablations (only if CellMem beats baseline) ---
    cellmem_better = False
    bl_accs, cm_accs = [], []
    for seed in args.seeds:
        bl_key = f"baseline/seed{seed}"
        cm_key = f"cellmem/seed{seed}"
        if bl_key in results and cm_key in results:
            bl_w = min(100, len(results[bl_key]['acc_history']))
            cm_w = min(100, len(results[cm_key]['acc_history']))
            bl_accs.append(sum(results[bl_key]['acc_history'][-bl_w:]) / bl_w)
            cm_accs.append(sum(results[cm_key]['acc_history'][-cm_w:]) / cm_w)
    if bl_accs and cm_accs:
        bl_mean = sum(bl_accs) / len(bl_accs)
        cm_mean = sum(cm_accs) / len(cm_accs)
        cellmem_better = cm_mean > bl_mean

    if not args.skip_ablations and cellmem_better:
        print("\n" + "=" * 60)
        print("ABLATION STUDY (CellMem beat baseline)")
        print("=" * 60)

        ablation_variants = [
            'ablation_no_topology',
            'ablation_no_astrocyte',
            'ablation_hebbian',
            'ablation_single_msb',
        ]
        ablation_results = {}
        # Use a single seed for ablations (faster)
        abl_seed = args.seeds[0]
        for variant in ablation_variants:
            model = make_model(variant, d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
                               n_keys=args.n_keys, n_values=args.n_values, n_pairs=args.n_pairs,
                               d_cell=args.d_cell, n_cells=args.n_cells)
            model = model.to(device)
            n_params = count_params(model)
            label = variant.replace('ablation_', '')
            print(f"\n--- Ablation: {label} | Params: {n_params:,} ---")

            metrics = train_and_evaluate(
                model_name=f"{variant}/seed{abl_seed}",
                model=model,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_pairs=args.n_pairs,
                n_keys=args.n_keys,
                n_values=args.n_values,
                lr=args.lr,
                device=device,
                seed=abl_seed,
            )
            ablation_results[variant] = metrics
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print("ABLATION SUMMARY (seed={})".format(abl_seed))
        print("=" * 60)
        # Include baseline and full cellmem for comparison
        ref_key_bl = f"baseline/seed{abl_seed}"
        ref_key_cm = f"cellmem/seed{abl_seed}"
        for label, metrics_dict in [("baseline", results.get(ref_key_bl)),
                                     ("cellmem (full)", results.get(ref_key_cm))]:
            if metrics_dict:
                w = min(100, len(metrics_dict['acc_history']))
                acc = sum(metrics_dict['acc_history'][-w:]) / w
                print(f"  {label:>25s}: {acc:.1%}")
        for variant in ablation_variants:
            label = variant.replace('ablation_', '')
            metrics_dict = ablation_results[variant]
            w = min(100, len(metrics_dict['acc_history']))
            acc = sum(metrics_dict['acc_history'][-w:]) / w
            print(f"  {label:>25s}: {acc:.1%}")
    elif not args.skip_ablations and not cellmem_better:
        print("\n[INFO] CellMem did not beat baseline; skipping ablations.")

    # --- Scaling test: vary n_pairs ---
    if not args.skip_scaling:
        print("\n" + "=" * 60)
        print("SCALING TEST — varying n_pairs")
        print("=" * 60)
        scaling_pairs = [4, 8, 16]
        # Use a limited seq_len check: n_pairs=32 needs seq_len=131 which exceeds 64
        # Adjust: only test pairs that fit within a reasonable seq_len
        max_seq_len = 256  # allow up to 256
        scaling_pairs = [np for np in scaling_pairs if (4 * np + 3) <= max_seq_len]
        scaling_results = {}
        scale_seed = args.seeds[0]

        for np in scaling_pairs:
            seq_len_np = 4 * np + 3
            for variant in ['baseline', 'cellmem']:
                model = make_model(variant, d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
                                   n_keys=args.n_keys, n_values=args.n_values, n_pairs=np,
                                   d_cell=args.d_cell, n_cells=args.n_cells)
                # Update positional embedding size for larger sequences
                if seq_len_np > model.config.seq_len:
                    model.config.seq_len = seq_len_np
                    model.pos_embedding = nn.Embedding(seq_len_np, model.config.d_model)
                    nn.init.normal_(model.pos_embedding.weight, mean=0.0, std=0.02)
                model = model.to(device)
                label = f"{variant}/np{np}"
                print(f"\n--- {label} (seq_len={seq_len_np}) ---")
                metrics = train_and_evaluate(
                    model_name=label,
                    model=model,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_pairs=np,
                    n_keys=args.n_keys,
                    n_values=args.n_values,
                    lr=args.lr,
                    device=device,
                    seed=scale_seed,
                )
                scaling_results[label] = metrics
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print("SCALING SUMMARY (seed={})".format(scale_seed))
        print("=" * 60)
        print(f"  {'n_pairs':>8s}  {'baseline':>10s}  {'cellmem':>10s}  {'delta':>8s}")
        print(f"  {'-------':>8s}  {'--------':>10s}  {'-------':>10s}  {'-----':>8s}")
        for np in scaling_pairs:
            bl_key = f"baseline/np{np}"
            cm_key = f"cellmem/np{np}"
            bl_m = scaling_results.get(bl_key)
            cm_m = scaling_results.get(cm_key)
            bl_acc = sum(bl_m['acc_history'][-100:]) / min(100, len(bl_m['acc_history'])) if bl_m else float('nan')
            cm_acc = sum(cm_m['acc_history'][-100:]) / min(100, len(cm_m['acc_history'])) if cm_m else float('nan')
            delta = cm_acc - bl_acc
            sign = '+' if delta >= 0 else ''
            print(f"  {np:>8d}  {bl_acc:>10.1%}  {cm_acc:>10.1%}  {sign}{delta:>7.1%}")

    print("\nDone.")


if __name__ == '__main__':
    main()
