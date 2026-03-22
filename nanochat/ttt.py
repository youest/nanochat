"""
TTT (Test-Time Training) layer module.
The "hidden state" is the weights of a small inner model (W_inner) that gets updated
via gradient descent on a self-supervised reconstruction loss during inference.
Baseline comparison for CellMem — matches CellMem's interface exactly.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class TTTConfig:
    d_model: int = 768
    d_cell: int = 32
    n_cells: int = 2
    lr_init: float = 0.01  # inner learning rate


class TTTLayer(nn.Module):
    def __init__(self, config: TTTConfig):
        super().__init__()
        self.config = config
        d_model, d_cell, n_cells = config.d_model, config.d_cell, config.n_cells
        assert d_model % n_cells == 0
        d_slice = d_model // n_cells

        # Per-cell learnable parameters
        self.W_in = nn.ParameterList([nn.Parameter(torch.randn(d_slice, d_cell) * 0.02) for _ in range(n_cells)])
        self.W_msb = nn.ModuleList([
            nn.ParameterList([nn.Parameter(torch.randn(d_cell, d_slice) * 0.02) for _ in range(4)])
            for _ in range(n_cells)
        ])
        self.lr_param = nn.ParameterList([nn.Parameter(torch.tensor(config.lr_init)) for _ in range(n_cells)])

        # Runtime state (not parameters, set by reset_state)
        self._W_inner = None

    def reset_state(self, batch_size: int):
        d_cell, n_cells = self.config.d_cell, self.config.n_cells
        # Initialize W_inner to small identity-like values
        self._W_inner = [torch.eye(d_cell) * 0.01 for _ in range(n_cells)]

    def get_state(self):
        return {
            "W_inner": [w.detach().clone() for w in self._W_inner],
        }

    def forward(self, x):
        B, d_model = x.shape
        n_cells = self.config.n_cells
        d_cell = self.config.d_cell
        d_slice = d_model // n_cells

        out_attn, out_mlp, out_add, out_x0 = [], [], [], []

        for i in range(n_cells):
            x_slice = x[:, i * d_slice:(i + 1) * d_slice]  # (B, d_slice)

            # 1. Project input to cell space
            z = x_slice @ self.W_in[i]  # (B, d_cell)

            # 2. Inner model prediction — W_inner IS the memory/state
            W_inner = self._W_inner[i]
            W_inner_param = W_inner.detach().requires_grad_(True)
            z_d = z.detach()
            z_pred = z_d @ W_inner_param  # (B, d_cell)

            # 3. Self-supervised loss: reconstruct z from z_pred
            loss_inner = ((z_d - z_pred) ** 2).sum(dim=-1).mean()  # scalar

            # 4. Gradient step on W_inner (detached from outer autograd graph)
            grad = torch.autograd.grad(loss_inner, W_inner_param, create_graph=False)[0]
            lr = self.lr_param[i].detach()
            W_inner_new = (W_inner_param - lr * grad).detach()
            self._W_inner[i] = W_inner_new

            # 5. Output: use updated W_inner for MSB fan-out
            # z still has gradients w.r.t. W_in (from step 1) — this is intentional
            mem_out = z @ W_inner_new  # (B, d_cell) — W_inner_new is detached, gradients flow through z

            # 6. MSB fan-out (same as CellMem)
            out_attn.append(mem_out @ self.W_msb[i][0])   # (B, d_slice)
            out_mlp.append(mem_out @ self.W_msb[i][1])     # (B, d_slice)
            out_add.append(mem_out @ self.W_msb[i][2])     # (B, d_slice)
            out_x0.append(mem_out @ self.W_msb[i][3])      # (B, d_slice)

        g_attn = torch.cat(out_attn, dim=-1)
        g_mlp = torch.cat(out_mlp, dim=-1)
        r_add = torch.cat(out_add, dim=-1)
        x0_mod = torch.cat(out_x0, dim=-1)
        return g_attn, g_mlp, r_add, x0_mod
