"""
Test TTT (Test-Time Training) module. Example run:

python -m pytest tests/test_ttt.py -v
"""

import torch
import pytest
from nanochat.ttt import TTTLayer, TTTConfig


# Small config for fast tests
CFG = TTTConfig(d_model=16, d_cell=8, n_cells=2)


class TestTTTLayer:
    """Tests for the TTT layer module."""

    def test_output_shapes(self):
        """All 4 MSB outputs should have shape (B, d_model)."""
        torch.manual_seed(0)
        ttt = TTTLayer(CFG)
        B = 4
        ttt.reset_state(B)
        x = torch.randn(B, CFG.d_model)
        g_attn, g_mlp, r_add, x0_mod = ttt(x)

        for name, out in [("g_attn", g_attn), ("g_mlp", g_mlp), ("r_add", r_add), ("x0_mod", x0_mod)]:
            assert out.shape == (B, CFG.d_model), f"{name} shape {out.shape} != ({B}, {CFG.d_model})"

    def test_state_updates(self):
        """W_inner should change after forward passes."""
        torch.manual_seed(0)
        ttt = TTTLayer(CFG)
        ttt.reset_state(1)

        state_before = ttt.get_state()
        W_before = [w.clone() for w in state_before["W_inner"]]

        x = torch.randn(1, CFG.d_model)
        ttt(x)

        state_after = ttt.get_state()
        W_after = state_after["W_inner"]

        changed = any(not torch.allclose(wb, wa) for wb, wa in zip(W_before, W_after))
        assert changed, "W_inner should change after a forward pass"

    def test_reset_clears_state(self):
        """After reset, W_inner should be restored to initial values."""
        torch.manual_seed(0)
        ttt = TTTLayer(CFG)
        ttt.reset_state(1)

        # Save initial state
        state_initial = ttt.get_state()
        W_initial = [w.clone() for w in state_initial["W_inner"]]

        # Run several forward passes to modify state
        x = torch.randn(1, CFG.d_model)
        for _ in range(10):
            ttt(x)

        # State should be different from initial
        state_modified = ttt.get_state()
        assert any(not torch.allclose(wi, wm) for wi, wm in zip(W_initial, state_modified["W_inner"])), \
            "W_inner should be modified after forward passes"

        # Reset and verify state is back to initial pattern (identity-like * 0.01)
        ttt.reset_state(1)
        state_reset = ttt.get_state()
        d_cell = CFG.d_cell
        expected = torch.eye(d_cell) * 0.01
        for w in state_reset["W_inner"]:
            assert torch.allclose(w, expected), "W_inner should be reset to initial identity-like values"

    def test_gradient_flows(self):
        """Gradients should flow through W_in and W_msb for training."""
        torch.manual_seed(0)
        ttt = TTTLayer(CFG)
        ttt.reset_state(1)
        x = torch.randn(1, CFG.d_model, requires_grad=True)
        g_attn, g_mlp, r_add, x0_mod = ttt(x)
        loss = g_attn.sum() + g_mlp.sum() + r_add.sum() + x0_mod.sum()
        loss.backward()

        # Check gradients on learnable params
        for name, p in ttt.named_parameters():
            if "W_in" in name or "W_msb" in name:
                assert p.grad is not None, f"No gradient for {name}"
                assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_deterministic(self):
        """Same input sequence should produce same state with same seed."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            ttt = TTTLayer(CFG)
            ttt.reset_state(1)
            torch.manual_seed(99)
            for _ in range(20):
                x = torch.randn(1, CFG.d_model)
                ttt(x)
            state = ttt.get_state()
            results.append([w.clone() for w in state["W_inner"]])

        for w1, w2 in zip(results[0], results[1]):
            assert torch.allclose(w1, w2), "Same seed should produce same state"

    def test_reconstruction_improves(self):
        """After many passes of the same pattern, inner reconstruction loss should decrease."""
        torch.manual_seed(0)
        ttt = TTTLayer(CFG)
        ttt.reset_state(1)

        # Fixed input pattern
        torch.manual_seed(7)
        x = torch.randn(1, CFG.d_model)

        d_slice = CFG.d_model // CFG.n_cells
        losses = []
        for _ in range(100):
            # Compute inner reconstruction loss manually before forward
            total_loss = 0.0
            for i in range(CFG.n_cells):
                x_slice = x[:, i * d_slice:(i + 1) * d_slice]
                z = x_slice @ ttt.W_in[i]  # (1, d_cell)
                z_d = z.detach()
                W_inner = ttt._W_inner[i]
                z_pred = z_d @ W_inner
                cell_loss = ((z_d - z_pred) ** 2).sum(dim=-1).mean().item()
                total_loss += cell_loss
            losses.append(total_loss)
            ttt(x)

        # Compare early vs late losses
        early_loss = sum(losses[:5]) / 5
        late_loss = sum(losses[-5:]) / 5
        assert late_loss < early_loss, \
            f"Inner loss should decrease over time: early={early_loss:.6f} late={late_loss:.6f}"
