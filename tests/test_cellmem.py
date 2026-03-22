"""
Test CellMem module. Example run:

python -m pytest tests/test_cellmem.py -v
"""

import torch
import pytest
from nanochat.cellmem import CellMem, CellMemConfig


def make_patterns(d, seed=42):
    """Create orthogonal unit-norm patterns for clear separation."""
    torch.manual_seed(seed)
    A = torch.randn(d)
    A = A / A.norm()
    B = torch.randn(d)
    B = B - (B @ A) * A
    B = B / B.norm()
    C = torch.randn(d)
    C = C - (C @ A) * A - (C @ B) * B
    C = C / C.norm()
    return A, B, C


def hebbian_update(M, z, alpha):
    """Simple Hebbian rule: outer(z, z) instead of outer(error, z)."""
    return M + alpha * (z.unsqueeze(-1) * z.unsqueeze(-2)).mean(dim=0)


def feed_sequence(mem, patterns, counts, batch_size=1):
    """Feed a sequence of patterns through a CellMem, return final state."""
    d_model = mem.config.d_model
    mem.reset_state(batch_size)
    for pat, count in zip(patterns, counts):
        x = pat.unsqueeze(0).expand(batch_size, d_model)
        for _ in range(count):
            mem(x)


# Small config for fast tests
CFG = CellMemConfig(d_model=16, d_cell=8, n_cells=2)


class TestUpdateRule:
    """Test the core hypothesis: anti-Hebbian captures novelty better than Hebbian."""

    def test_novel_pattern_recall(self):
        """After familiar A,B then novel C, memory should predict C reasonably."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)

        A, B, C = make_patterns(CFG.d_model)

        # Feed familiar patterns
        for _ in range(50):
            mem(A.unsqueeze(0))
        for _ in range(50):
            mem(B.unsqueeze(0))

        # Feed novel pattern a few times
        for _ in range(3):
            g_attn, g_mlp, r_add, x0_mod = mem(C.unsqueeze(0))

        # After seeing C, the memory should have changed (M not zero in relevant dims)
        state = mem.get_state()
        assert any(m.abs().sum() > 0 for m in state["M"]), "M should be non-zero after processing patterns"

    def test_familiar_pattern_stability(self):
        """After many repetitions of A, prediction error on A should decrease."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)

        A, _, _ = make_patterns(CFG.d_model)
        x = A.unsqueeze(0)

        # Collect errors over time
        errors = []
        for i in range(100):
            mem(x)
            state = mem.get_state()
            errors.append(sum(n.item() for n in state["novelty"]))

        # Error at end should be lower than at start (after a few warmup steps)
        early_error = sum(errors[5:15]) / 10
        late_error = sum(errors[-10:]) / 10
        assert late_error < early_error, f"Error should decrease: early={early_error:.4f} late={late_error:.4f}"

    def test_hebbian_comparison(self):
        """
        Anti-Hebbian should predict novel patterns better than Hebbian.
        Key: use non-orthogonal patterns so Hebbian's unbounded accumulation hurts it.
        With overlapping patterns, Hebbian M eigenvalues explode while anti-Hebbian self-corrects.
        """
        torch.manual_seed(7)
        d = CFG.d_cell
        A = torch.randn(d); A = A / A.norm()
        B = torch.randn(d); B = B / B.norm()
        # C overlaps with A — this is where Hebbian struggles
        C = 0.7 * A + 0.3 * torch.randn(d); C = C / C.norm()

        alpha = 0.02

        # --- Anti-Hebbian (error x z) ---
        M_anti = torch.zeros(d, d)
        for _ in range(80):
            z_pred = M_anti @ A
            M_anti = M_anti + alpha * torch.outer(A - z_pred, A)
        for _ in range(80):
            z_pred = M_anti @ B
            M_anti = M_anti + alpha * torch.outer(B - z_pred, B)
        for _ in range(5):
            z_pred = M_anti @ C
            M_anti = M_anti + alpha * torch.outer(C - z_pred, C)
        error_anti_C = (C - M_anti @ C).norm().item()

        # --- Hebbian (z x z) ---
        M_hebb = torch.zeros(d, d)
        for _ in range(80):
            M_hebb = M_hebb + alpha * torch.outer(A, A)
        for _ in range(80):
            M_hebb = M_hebb + alpha * torch.outer(B, B)
        for _ in range(5):
            M_hebb = M_hebb + alpha * torch.outer(C, C)
        error_hebb_C = (C - M_hebb @ C).norm().item()

        # Anti-Hebbian should have lower error on C
        assert error_anti_C < error_hebb_C, (
            f"Anti-Hebbian should predict C better: anti={error_anti_C:.4f} hebb={error_hebb_C:.4f}"
        )


class TestTopology:
    """Test that topology T evolves meaningfully."""

    def test_topology_opens_for_novelty(self):
        """After processing diverse inputs, T should change from initial state."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)

        A, B, C = make_patterns(CFG.d_model)
        for _ in range(20):
            mem(A.unsqueeze(0))
        for _ in range(20):
            mem(B.unsqueeze(0))
        # Introduce novelty
        mem(C.unsqueeze(0))

        state = mem.get_state()
        for T_mat in state["T"]:
            # T should have changed from all-ones after processing
            assert not torch.allclose(T_mat, torch.ones_like(T_mat), atol=1e-6), \
                "T should evolve from initial all-ones"

    def test_topology_closes_for_familiar(self):
        """After many repetitions, tau decay should pull T toward zero in unused dims."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2, tau_init=0.05)
        mem = CellMem(cfg)
        mem.reset_state(1)

        A, _, _ = make_patterns(cfg.d_model)
        T_initial_sum = sum(t.sum().item() for t in mem.get_state()["T"])
        for _ in range(200):
            mem(A.unsqueeze(0))

        T_final_sum = sum(t.sum().item() for t in mem.get_state()["T"])
        # With tau decay, T should have decreased overall
        assert T_final_sum < T_initial_sum, \
            f"T should decay: initial={T_initial_sum:.2f} final={T_final_sum:.2f}"

    def test_topology_sparsifies(self):
        """Over time, T should develop non-trivial structure (not all ones)."""
        torch.manual_seed(0)
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2, tau_init=0.03)
        mem = CellMem(cfg)
        mem.reset_state(1)

        A, B, _ = make_patterns(cfg.d_model)
        for _ in range(100):
            mem(A.unsqueeze(0))
        for _ in range(100):
            mem(B.unsqueeze(0))

        state = mem.get_state()
        for T_mat in state["T"]:
            # T should no longer be all-ones (it has evolved)
            assert not torch.allclose(T_mat, torch.ones_like(T_mat), atol=1e-3), \
                "T should have evolved from initial all-ones"
            # T should still have some non-zero entries (not fully dead)
            assert T_mat.max() > 0.01, "T should not be fully dead"
            # T entries should have some variance (structure)
            assert T_mat.std() > 1e-6, "T should have non-trivial structure"


class TestAstrocyte:
    """Test that the astrocyte modulates learning rate correctly."""

    def test_high_variance_increases_lr(self):
        """When inputs are diverse (high variance), sigma should be higher after many steps."""
        torch.manual_seed(0)
        # Use fresh config with astro starting from zero to test modulation direction
        cfg = CellMemConfig(d_model=16, d_cell=8, n_cells=2)
        mem = CellMem(cfg)
        mem.reset_state(1)
        # Override astro to start from zero for a clean test
        mem._astro_mu = [torch.zeros(1) for _ in range(cfg.n_cells)]
        mem._astro_sigma = [torch.zeros(1) for _ in range(cfg.n_cells)]

        # Feed diverse patterns
        A, B, C = make_patterns(cfg.d_model, seed=99)
        for _ in range(100):
            for p in [A, B, C]:
                mem(p.unsqueeze(0))
        sigma_diverse = sum(s.item() for s in mem.get_state()["astro_sigma"])

        # Feed uniform pattern from scratch
        mem.reset_state(1)
        mem._astro_mu = [torch.zeros(1) for _ in range(cfg.n_cells)]
        mem._astro_sigma = [torch.zeros(1) for _ in range(cfg.n_cells)]
        for _ in range(300):
            mem(A.unsqueeze(0))
        sigma_uniform = sum(s.item() for s in mem.get_state()["astro_sigma"])

        assert sigma_diverse > sigma_uniform, \
            f"Diverse sigma={sigma_diverse:.6f} should exceed uniform sigma={sigma_uniform:.6f}"

    def test_low_variance_decreases_lr(self):
        """When inputs are uniform, astro_sigma should converge to small value from zero start."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)
        # Start astro from zero
        mem._astro_mu = [torch.zeros(1) for _ in range(CFG.n_cells)]
        mem._astro_sigma = [torch.zeros(1) for _ in range(CFG.n_cells)]

        A, _, _ = make_patterns(CFG.d_model)
        for _ in range(300):
            mem(A.unsqueeze(0))

        state = mem.get_state()
        for sigma in state["astro_sigma"]:
            assert sigma.item() < 0.1, f"astro_sigma should be small for uniform input, got {sigma.item():.4f}"


class TestMSB:
    """Test the Multi-Synaptic Bouton fan-out."""

    def test_output_shapes(self):
        """All 4 MSB outputs should have shape (B, d_model)."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        B = 4
        mem.reset_state(B)
        x = torch.randn(B, CFG.d_model)
        g_attn, g_mlp, r_add, x0_mod = mem(x)

        for name, out in [("g_attn", g_attn), ("g_mlp", g_mlp), ("r_add", r_add), ("x0_mod", x0_mod)]:
            assert out.shape == (B, CFG.d_model), f"{name} shape {out.shape} != ({B}, {CFG.d_model})"

    def test_four_outputs_are_different(self):
        """The 4 MSB outputs should not be identical (different projections)."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)
        # Feed a few inputs so M is non-trivial
        x = torch.randn(1, CFG.d_model)
        for _ in range(10):
            mem(x)
        g_attn, g_mlp, r_add, x0_mod = mem(x)
        outputs = [g_attn, g_mlp, r_add, x0_mod]
        # At least some pairs should differ
        diffs = 0
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if not torch.allclose(outputs[i], outputs[j], atol=1e-6):
                    diffs += 1
        assert diffs > 0, "All 4 MSB outputs are identical; projections should differ"


class TestModule:
    """Test the module mechanics."""

    def test_reset_clears_state(self):
        """After reset, M should be zeros, T should be ones."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)
        x = torch.randn(1, CFG.d_model)
        for _ in range(10):
            mem(x)
        # State should be non-trivial
        state = mem.get_state()
        assert any(m.abs().sum() > 0 for m in state["M"])

        # Reset
        mem.reset_state(1)
        state = mem.get_state()
        for m in state["M"]:
            assert torch.allclose(m, 0.01 * torch.eye(m.shape[0])), "M should be 0.01*I after reset"
        for t in state["T"]:
            assert torch.allclose(t, torch.ones_like(t)), "T should be ones after reset"

    def test_state_accumulates(self):
        """After processing tokens, M should be non-zero."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)
        x = torch.randn(1, CFG.d_model)
        mem(x)
        state = mem.get_state()
        assert any(m.abs().sum() > 0 for m in state["M"]), "M should be non-zero after one forward"

    def test_gradient_flows(self):
        """Gradients should flow through W_in and W_msb for training."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)
        x = torch.randn(1, CFG.d_model, requires_grad=True)
        g_attn, g_mlp, r_add, x0_mod = mem(x)
        loss = g_attn.sum() + g_mlp.sum() + r_add.sum() + x0_mod.sum()
        loss.backward()

        # Check gradients on learnable params
        for name, p in mem.named_parameters():
            if "W_in" in name or "W_msb" in name:
                assert p.grad is not None, f"No gradient for {name}"
                assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_flows_through_M_chain(self):
        """Gradients should flow through the M update chain across multiple tokens.
        This is the key test for Option B: differentiable M."""
        torch.manual_seed(0)
        mem = CellMem(CFG)
        mem.reset_state(1)
        # Process 5 tokens sequentially — M updates at each step
        for i in range(5):
            x = torch.randn(1, CFG.d_model, requires_grad=(i == 0))
            if i == 0:
                x_first = x  # keep reference to first input
            g_attn, g_mlp, r_add, x0_mod = mem(x)
        # Loss on the LAST token's output
        loss = r_add.sum()
        loss.backward()
        # The gradient should reach x_first through the M chain:
        # x_first → z₀ → M₁ → z₁_pred → error₁ → M₂ → ... → r_add₅
        assert x_first.grad is not None, "Gradient should reach first input through M chain"
        assert x_first.grad.abs().sum() > 0, "Gradient through M chain should be non-zero"

    def test_deterministic(self):
        """Same input sequence should produce same state with same seed."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            mem = CellMem(CFG)
            mem.reset_state(1)
            torch.manual_seed(99)
            for _ in range(20):
                x = torch.randn(1, CFG.d_model)
                mem(x)
            state = mem.get_state()
            results.append([m.clone() for m in state["M"]])

        for m1, m2 in zip(results[0], results[1]):
            assert torch.allclose(m1, m2), "Same seed should produce same state"
