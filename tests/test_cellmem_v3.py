# tests/test_cellmem_v3.py
import torch
import pytest
from nanochat.cellmem_v3 import CellMemConfig
from nanochat.kv_interceptor import KVInterceptor


class TestCellMemConfigV3:
    def test_default_values(self):
        cfg = CellMemConfig()
        assert cfg.n_slots == 256
        assert cfg.router_dim == 128
        assert cfg.router_layers is None  # auto top-4
        assert cfg.episode_size == 8
        assert cfg.top_k == 4
        assert cfg.contrastive_tau == 0.07
        assert cfg.surprise_threshold == 4.0
        assert cfg.decay_factor == 0.95

    def test_custom_router_layers(self):
        cfg = CellMemConfig(router_layers=[20, 25, 30, 35])
        assert cfg.router_layers == [20, 25, 30, 35]


class TestKVInterceptor:
    def _make_mock_model(self):
        """Create a minimal model with attention layers that have k_proj/v_proj."""
        class FakeAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = torch.nn.Linear(32, 16, bias=False)
                self.v_proj = torch.nn.Linear(32, 16, bias=False)
                self.rotary_emb = None

            def forward(self, x):
                return x

        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

            def forward(self, x):
                return x

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([FakeLayer() for _ in range(4)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class FakeWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeModel()

        return FakeWrapper()

    def test_register_and_capture(self):
        model = self._make_mock_model()
        interceptor = KVInterceptor(model, layer_indices=[2, 3], model_type="qwen")
        x = torch.randn(1, 10, 32)
        model.model(x)
        buf = interceptor.get_buffered_kv()
        assert 2 in buf
        assert 3 in buf
        assert buf[2][0].shape[-1] == 16

    def test_clear_buffer(self):
        model = self._make_mock_model()
        interceptor = KVInterceptor(model, layer_indices=[2], model_type="qwen")
        x = torch.randn(1, 5, 32)
        model.model(x)
        interceptor.clear_buffer()
        assert len(interceptor.get_buffered_kv()) == 0

    def test_remove_hooks(self):
        model = self._make_mock_model()
        interceptor = KVInterceptor(model, layer_indices=[2], model_type="qwen")
        interceptor.remove_hooks()
        x = torch.randn(1, 5, 32)
        model.model(x)
        assert len(interceptor.get_buffered_kv()) == 0


from nanochat.cellmem_v3 import MemoryRouter


class TestMemoryRouter:
    def test_init_nonzero_weights(self):
        router = MemoryRouter(d_model=64, d_router=16)
        assert router.w_q_r.weight.abs().sum() > 0
        assert router.w_k_r.weight.abs().sum() > 0

    def test_route_returns_topk_indices(self):
        router = MemoryRouter(d_model=64, d_router=16, top_k=2)
        h_query = torch.randn(1, 10, 64)  # [B, T, d_model]
        router_keys = torch.randn(8, 16)   # [n_episodes, d_router]
        indices, scores = router.route(h_query, router_keys)
        assert indices.shape == (1, 10, 2)  # [B, T, top_k]
        assert scores.shape == (1, 10, 2)

    def test_route_empty_memory_returns_empty(self):
        router = MemoryRouter(d_model=64, d_router=16, top_k=2)
        h_query = torch.randn(1, 10, 64)
        router_keys = torch.zeros(0, 16)
        indices, scores = router.route(h_query, router_keys)
        assert indices.shape[-1] == 0

    def test_route_fewer_episodes_than_topk(self):
        router = MemoryRouter(d_model=64, d_router=16, top_k=4)
        h_query = torch.randn(1, 5, 64)
        router_keys = torch.randn(2, 16)  # only 2 episodes, top_k=4
        indices, scores = router.route(h_query, router_keys)
        assert indices.shape == (1, 5, 2)  # min(top_k, n_episodes)

    def test_encode_episode_shape(self):
        router = MemoryRouter(d_model=64, d_router=16)
        h_tokens = torch.randn(8, 64)  # [episode_size, d_model]
        key = router.encode_episode(h_tokens)
        assert key.shape == (16,)  # [d_router]

    def test_encode_episode_partial(self):
        router = MemoryRouter(d_model=64, d_router=16)
        h_tokens = torch.randn(3, 64)  # partial episode, < 8
        key = router.encode_episode(h_tokens)
        assert key.shape == (16,)

    def test_contrastive_loss_positive(self):
        router = MemoryRouter(d_model=64, d_router=16)
        q = torch.randn(4, 16)       # [B, d_router]
        pos = q.clone()               # identical = perfect match
        negs = torch.randn(4, 3, 16)  # [B, n_neg, d_router]
        loss = router.contrastive_loss(q, pos, negs, tau=0.07)
        assert loss.item() >= 0
        assert loss.item() < 1.0  # should be low for identical pos

    def test_contrastive_loss_gradient_flows(self):
        router = MemoryRouter(d_model=64, d_router=16)
        h_query = torch.randn(4, 64, requires_grad=True)
        q = router.w_q_r(h_query)     # [B, d_router]
        pos = torch.randn(4, 16)
        negs = torch.randn(4, 3, 16)
        loss = router.contrastive_loss(q, pos, negs, tau=0.07)
        loss.backward()
        assert router.w_q_r.weight.grad is not None
        assert router.w_q_r.weight.grad.abs().sum() > 0
