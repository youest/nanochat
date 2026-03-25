"""Tests for CellMemWrapper components (no GPU required)."""
import torch
import torch.nn.functional as F
import pytest


class TestCellMemWrapperUsesContentGate:
    def test_content_gate_produces_per_token_values(self):
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=64)
        h_local = torch.randn(2, 8, 64)
        h_mem = torch.randn(2, 8, 64)
        g = gate(h_local, h_mem)
        assert g.shape == (2, 8, 1)

    def test_content_gate_trainable_param_count(self):
        from nanochat.cellmem_v2 import ContentGate
        gate = ContentGate(d_model=256)
        n_params = sum(p.numel() for p in gate.parameters())
        assert n_params > 1000
        assert n_params < 200_000

    def test_rms_norm_equalizes_attention(self):
        from nanochat.cellmem_v2 import MemoryRMSNorm
        d = 64
        norm = MemoryRMSNorm(d_model=d)
        mem = torch.randn(5, d)
        mem[0] *= 100
        query = torch.randn(1, d)
        scores_raw = (query @ mem.T) / (d ** 0.5)
        attn_raw = torch.softmax(scores_raw, dim=-1)
        # clamp to avoid log(0)=-inf producing nan entropy
        entropy_raw = -(attn_raw * attn_raw.clamp(min=1e-9).log()).sum()
        mem_normed = norm(mem)
        scores_normed = (query @ mem_normed.T) / (d ** 0.5)
        attn_normed = torch.softmax(scores_normed, dim=-1)
        entropy_normed = -(attn_normed * attn_normed.clamp(min=1e-9).log()).sum()
        assert entropy_normed > entropy_raw


class TestMixedTrainingData:
    def test_negative_data_has_irrelevant_memory(self):
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=50, seed=42)
        negatives = [d for d in data if d["type"] == "negative"]
        assert len(negatives) > 0
        for neg in negatives:
            assert "context" in neg
            assert "query" in neg
            assert "answer" in neg
            assert neg["type"] == "negative"

    def test_poisoned_data_has_wrong_memory(self):
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=50, seed=42)
        poisoned = [d for d in data if d["type"] == "poisoned"]
        assert len(poisoned) > 0
        for p in poisoned:
            assert "context" in p
            assert "query" in p
            assert "answer" in p
            assert "wrong_context" in p

    def test_data_ratios(self):
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=100, seed=42)
        counts = {"positive": 0, "negative": 0, "poisoned": 0}
        for d in data:
            counts[d["type"]] += 1
        assert counts["positive"] >= 30
        assert counts["negative"] >= 30
        assert counts["poisoned"] >= 10

    def test_positive_data_matches_original_format(self):
        from scripts.train_cellmem_qwen import _generate_mixed_data
        data = _generate_mixed_data(n=20, seed=42)
        positives = [d for d in data if d["type"] == "positive"]
        assert len(positives) > 0
        for p in positives:
            assert "context" in p
            assert "query" in p
            assert "answer" in p


class TestGateSupervisionLoss:
    def test_compute_gate_loss_negatives(self):
        from scripts.train_cellmem_qwen import compute_gate_loss
        gate_values = torch.tensor([[[0.8]], [[0.6]], [[0.9]]])
        loss = compute_gate_loss(gate_values, target="close")
        assert loss.shape == ()
        assert loss.item() > 0.5

    def test_compute_gate_loss_positives(self):
        from scripts.train_cellmem_qwen import compute_gate_loss
        gate_values = torch.tensor([[[0.2]], [[0.3]], [[0.1]]])
        loss = compute_gate_loss(gate_values, target="open")
        assert loss.shape == ()
        assert loss.item() > 0.5

    def test_gate_loss_zero_when_correct(self):
        from scripts.train_cellmem_qwen import compute_gate_loss
        gate_close = torch.tensor([[[0.01]], [[0.02]]])
        loss_close = compute_gate_loss(gate_close, target="close")
        assert loss_close.item() < 0.05
        gate_open = torch.tensor([[[0.98]], [[0.99]]])
        loss_open = compute_gate_loss(gate_open, target="open")
        assert loss_open.item() < 0.05

    def test_gate_loss_gradient_flows(self):
        from scripts.train_cellmem_qwen import compute_gate_loss
        gate_values = torch.tensor([[[0.7]]], requires_grad=True)
        loss = compute_gate_loss(gate_values, target="close")
        loss.backward()
        assert gate_values.grad is not None
        assert gate_values.grad.abs().sum() > 0


class TestEvalMetrics:
    def test_eval_comprehensive_signature(self):
        from scripts.train_cellmem_qwen import eval_comprehensive
        import inspect
        sig = inspect.signature(eval_comprehensive)
        params = list(sig.parameters.keys())
        assert "wrapper" in params
        assert "tokenizer" in params
        assert "data" in params
        assert "device" in params

    def test_eval_comprehensive_docstring_mentions_all_metrics(self):
        from scripts.train_cellmem_qwen import eval_comprehensive
        doc = eval_comprehensive.__doc__
        assert doc is not None
        for key in ["positive_recall", "poisoned_resistance",
                     "multi_memory_recall", "mean_gate_positive", "mean_gate_negative"]:
            assert key in doc, f"Docstring missing metric: {key}"


class TestRunExperimentWiring:
    def test_n_examples_arg_exists(self):
        from scripts.train_cellmem_qwen import main
        import inspect
        source = inspect.getsource(main)
        assert 'n-examples' in source or 'n_examples' in source

    def test_run_experiment_calls_eval_comprehensive(self):
        import inspect
        from scripts.train_cellmem_qwen import run_experiment
        source = inspect.getsource(run_experiment)
        assert 'eval_comprehensive' in source
