"""
Test CellMem integration in GPT model.

python -m pytest tests/test_gpt_cellmem.py -v
"""

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig


def make_small_config(cellmem_enabled=False):
    """Small config for fast tests."""
    return GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
        cellmem_enabled=cellmem_enabled,
        cellmem_d_cell=16,
        cellmem_n_cells=2,
        cellmem_chunk_size=16,
        cellmem_layers=(0, 3),  # first and last layer
    )


def build_model(cellmem_enabled=False):
    cfg = make_small_config(cellmem_enabled)
    model = GPT(cfg, pad_vocab_size_to=1)
    model.init_weights()
    return model


class TestNoCellMem:
    """When cellmem_enabled=False, model should be identical to original."""

    def test_no_cellmem_modules(self):
        model = build_model(cellmem_enabled=False)
        assert len(model.cellmems) == 0

    def test_forward_shape(self):
        model = build_model(cellmem_enabled=False)
        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))
        loss = model(idx, targets)
        assert loss.shape == ()
        assert loss.item() > 0


class TestWithCellMem:
    """When cellmem_enabled=True, model should work with CellMem gates."""

    def test_cellmem_modules_created(self):
        model = build_model(cellmem_enabled=True)
        assert len(model.cellmems) == 2  # layers 0 and 3
        assert "0" in model.cellmems
        assert "3" in model.cellmems

    def test_forward_shape(self):
        model = build_model(cellmem_enabled=True)
        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))
        loss = model(idx, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flows_to_cellmem(self):
        model = build_model(cellmem_enabled=True)
        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))
        loss = model(idx, targets)
        loss.backward()
        # Check gradients on CellMem W_msb params (direct gradient path via gates)
        # W_in gradient may be zero in bf16 due to M=0.01*I chain
        for name, cm in model.cellmems.items():
            has_grad = False
            for p_name, p in cm.named_parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grad = True
                    break
            assert has_grad, f"No non-zero gradient in any cellmem[{name}] parameter"

    def test_r_add_x0_mod_start_at_zero(self):
        """r_add and x0_mod projections should be zero-initialized."""
        model = build_model(cellmem_enabled=True)
        for cm in model.cellmems.values():
            for i in range(cm.config.n_cells):
                assert torch.allclose(cm.W_msb[i][2], torch.zeros_like(cm.W_msb[i][2])), "r_add should be zero"
                assert torch.allclose(cm.W_msb[i][3], torch.zeros_like(cm.W_msb[i][3])), "x0_mod should be zero"


class TestOptimizer:
    """CellMem params should be in the correct optimizer groups."""

    def test_all_params_accounted(self):
        """setup_optimizer should not crash and account for all params."""
        model = build_model(cellmem_enabled=True)
        model.init_weights()
        optimizer = model.setup_optimizer()
        # Count params in optimizer
        opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
        model_params = len(list(model.parameters()))
        assert opt_params == model_params, f"Optimizer has {opt_params} params, model has {model_params}"


class TestGenerate:
    """Generate should work with CellMem enabled."""

    def test_generate_produces_tokens(self):
        model = build_model(cellmem_enabled=True)
        tokens = list(model.generate([1, 2, 3], max_tokens=5, temperature=1.0))
        assert len(tokens) == 5
        assert all(isinstance(t, int) for t in tokens)
