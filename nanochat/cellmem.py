"""
CellMem: biologically-inspired cellular memory module.
Each cell maintains a memory matrix M and topology mask T that self-update during inference.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class CellMemConfig:
    d_model: int = 768
    d_cell: int = 32
    n_cells: int = 2
    alpha_init: float = 0.1
    gamma_init: float = 0.01
    tau_init: float = 0.01


class CellMem(nn.Module):
    def __init__(self, config: CellMemConfig):
        super().__init__()
        self.config = config
        d_model, d_cell, n_cells = config.d_model, config.d_cell, config.n_cells
        assert d_model % n_cells == 0
        d_slice = d_model // n_cells

        # Per-cell learnable parameters (Xavier init for proper signal magnitude)
        self.W_in = nn.ParameterList([nn.Parameter(torch.randn(d_slice, d_cell) / d_slice**0.5) for _ in range(n_cells)])
        self.W_msb = nn.ModuleList([
            nn.ParameterList([nn.Parameter(torch.randn(d_cell, d_slice) / d_cell**0.5) for _ in range(4)])
            for _ in range(n_cells)
        ])
        self.alpha_base = nn.ParameterList([nn.Parameter(torch.tensor(config.alpha_init)) for _ in range(n_cells)])
        self.gamma = nn.ParameterList([nn.Parameter(torch.tensor(config.gamma_init)) for _ in range(n_cells)])
        self.tau = nn.ParameterList([nn.Parameter(torch.tensor(config.tau_init)) for _ in range(n_cells)])

        # Runtime state (not parameters, set by reset_state)
        self._M = None
        self._T = None
        self._astro_mu = None
        self._astro_sigma = None
        self._novelty = None

    def reset_state(self, batch_size: int):
        d_cell, n_cells = self.config.d_cell, self.config.n_cells
        self._M = [0.01 * torch.eye(d_cell) for _ in range(n_cells)]
        self._T = [torch.ones(d_cell, d_cell) for _ in range(n_cells)]
        self._astro_mu = [torch.ones(1) for _ in range(n_cells)]
        self._astro_sigma = [torch.ones(1) for _ in range(n_cells)]
        self._novelty = [torch.zeros(1) for _ in range(n_cells)]

    def get_state(self):
        return {
            "M": [m.detach().clone() for m in self._M],
            "T": [t.detach().clone() for t in self._T],
            "astro_mu": [mu.detach().clone() for mu in self._astro_mu],
            "astro_sigma": [sigma.detach().clone() for sigma in self._astro_sigma],
            "novelty": [n.detach().clone() for n in self._novelty],
        }

    def forward(self, x):
        B, d_model = x.shape
        n_cells = self.config.n_cells
        d_slice = d_model // n_cells

        out_attn, out_mlp, out_add, out_x0 = [], [], [], []

        for i in range(n_cells):
            x_slice = x[:, i * d_slice:(i + 1) * d_slice]  # (B, d_slice)

            # 1. Project input to cell space
            z = x_slice @ self.W_in[i]  # (B, d_cell)

            # 2. Memory prediction
            M, T = self._M[i], self._T[i]
            z_pred = (M * T) @ z.T  # (d_cell, B)
            z_pred = z_pred.T  # (B, d_cell)

            # 3. Compute error and novelty
            error = z - z_pred
            z_norm_sq = z.norm(dim=-1, keepdim=True) ** 2 + 1e-8
            novelty = (error.norm(dim=-1, keepdim=True) ** 2) / z_norm_sq
            self._novelty[i] = novelty.mean().detach()

            # 4. Astrocyte modulation
            z_norms = z.norm(dim=-1)  # (B,)
            mu = self._astro_mu[i]
            sigma = self._astro_sigma[i]
            mu = 0.99 * mu + 0.01 * z_norms.mean().detach()
            sigma = 0.99 * sigma + 0.01 * ((z_norms.detach() - mu) ** 2).mean()
            self._astro_mu[i] = mu
            self._astro_sigma[i] = sigma
            alpha_eff = self.alpha_base[i] * (sigma / (mu + 1e-8))

            # 5. Anti-Hebbian memory update (differentiable — gradients flow through M chain)
            delta_M = alpha_eff * (error.unsqueeze(-1) * z.unsqueeze(-2)).mean(dim=0)
            self._M[i] = M + delta_M

            # 6. Topology update (detached — T is a structural mask, not a smooth function)
            error_d = error.detach()
            z_d = z.detach()
            gamma_val = self.gamma[i].detach()
            tau_val = self.tau[i].detach()
            delta_T = gamma_val * (
                (error_d.abs().unsqueeze(-1) * z_d.abs().unsqueeze(-2)).mean(dim=0) - tau_val * T
            )
            self._T[i] = (T + delta_T).clamp(0, 1)

            # 7. MSB fan-out
            mem_out = (self._M[i] * self._T[i]) @ z.T  # (d_cell, B)
            mem_out = mem_out.T  # (B, d_cell)
            out_attn.append(mem_out @ self.W_msb[i][0])   # (B, d_slice)
            out_mlp.append(mem_out @ self.W_msb[i][1])     # (B, d_slice)
            out_add.append(mem_out @ self.W_msb[i][2])     # (B, d_slice)
            out_x0.append(mem_out @ self.W_msb[i][3])      # (B, d_slice)

        g_attn = torch.cat(out_attn, dim=-1)
        g_mlp = torch.cat(out_mlp, dim=-1)
        r_add = torch.cat(out_add, dim=-1)
        x0_mod = torch.cat(out_x0, dim=-1)
        return g_attn, g_mlp, r_add, x0_mod
