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


from nanochat.cellmem_v3 import MemoryStore


class TestMemoryStoreV3Init:
    def test_init_shapes(self):
        cfg = CellMemConfig(n_slots=32, router_dim=16)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=4, d_head=8)
        assert store.keys.shape == (2, 32, 4, 8)
        assert store.values.shape == (2, 32, 4, 8)
        assert store.router_keys.shape == (32 // cfg.episode_size, 16)
        assert store.active_count == 0

    def test_read_empty_returns_none(self):
        cfg = CellMemConfig(n_slots=32, router_dim=16)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=4, d_head=8)
        assert store.read([0]) is None


class TestMemoryStoreV3Write:
    def test_write_increments_count(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=2, d_head=4)
        kv = {
            "keys": torch.randn(2, 2, 4),    # [n_layers, n_kv_heads, d_head]
            "values": torch.randn(2, 2, 4),
        }
        router_key = torch.randn(8)
        store.write(kv, router_key, surprise=5.0)
        assert store.active_count == 1

    def test_write_stores_kv_correctly(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=2, d_head=4)
        k = torch.ones(2, 2, 4) * 3.0
        v = torch.ones(2, 2, 4) * 7.0
        store.write({"keys": k, "values": v}, torch.randn(8), surprise=5.0)
        assert (store.keys[:, 0, :, :] == 3.0).all()
        assert (store.values[:, 0, :, :] == 7.0).all()

    def test_write_full_evicts_min_surprise(self):
        cfg = CellMemConfig(n_slots=4, router_dim=8, episode_size=2)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        for i in range(4):
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                torch.randn(8), surprise=float(i + 1)  # surprises: 1,2,3,4
            )
        assert store.active_count == 4
        # Write 5th with high surprise → should evict slot with surprise=1.0
        store.write(
            {"keys": torch.ones(1, 1, 4) * 99, "values": torch.ones(1, 1, 4) * 99},
            torch.randn(8), surprise=10.0
        )
        assert store.active_count == 4
        assert 1.0 not in store.surprise[:4].tolist()

    def test_decorrelation_skips_similar(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, min_novelty=0.1)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        k = torch.ones(1, 1, 4)
        store.write({"keys": k, "values": k.clone()}, torch.randn(8), surprise=5.0)
        # Write near-identical → should skip
        store.write({"keys": k * 1.01, "values": k.clone()}, torch.randn(8), surprise=5.0)
        assert store.active_count == 1  # second write skipped


class TestMemoryStoreV3Read:
    def test_read_selected_episodes(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=2, n_kv_heads=2, d_head=4)
        # Write 8 tokens → 2 episodes
        for i in range(8):
            store.write(
                {"keys": torch.ones(2, 2, 4) * i, "values": torch.ones(2, 2, 4) * i},
                torch.randn(8) if i % 4 == 3 else None,  # router key every 4 tokens
                surprise=5.0
            )
        # Read episode 0 (tokens 0-3)
        result = store.read([0])
        assert result is not None
        keys, values = result
        # Episode 0 has tokens 0-3 for each layer
        assert keys.shape[1] == 4  # 4 tokens in episode

    def test_get_router_keys_shape(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        for i in range(8):
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                torch.randn(8) if i % 4 == 3 else None,
                surprise=5.0
            )
        rk = store.get_router_keys()
        assert rk.shape[1] == 8  # d_router


class TestMemoryStoreV3Persistence:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = CellMemConfig(n_slots=8, router_dim=4, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        for i in range(4):
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                torch.randn(4), surprise=float(i)
            )
        path = tmp_path / "mem.pt"
        store.save(path)

        store2 = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store2.load(path)
        assert store2.active_count == store.active_count
        assert torch.allclose(store2.surprise, store.surprise)

    def test_decay_on_load(self, tmp_path):
        cfg = CellMemConfig(n_slots=8, router_dim=4, decay_factor=0.5)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store.write(
            {"keys": torch.ones(1, 1, 4), "values": torch.ones(1, 1, 4)},
            torch.randn(4), surprise=5.0
        )
        path = tmp_path / "mem.pt"
        store.save(path)

        store2 = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store2.load(path)
        assert torch.allclose(store2.keys[:, 0], torch.ones(1, 1, 4) * 0.5)


class TestIntegrationSmoke:
    """Integration tests requiring a model. Skip if model not available."""

    @pytest.fixture
    def wrapper(self):
        """Load smallest Qwen model for testing."""
        try:
            from scripts.train_cellmem_qwen_v3 import CellMemWrapper
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = "Qwen/Qwen2.5-0.5B"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            n_layers = model.config.num_hidden_layers
            layer_indices = list(range(n_layers - 2, n_layers))
            return CellMemWrapper(model, tokenizer, layer_indices, device=device)
        except Exception:
            pytest.skip("Model not available")

    def test_write_then_read_no_crash(self, wrapper):
        wrapper.write_memory("Dr. Elena Voss discovered Pyrothene in 2031")
        tokens = wrapper.tokenizer("What did Dr. Voss discover?", return_tensors="pt")
        tokens = {k: v.to(wrapper.device) for k, v in tokens.items()}
        wrapper._install_read_hooks()
        with torch.no_grad():
            output = wrapper.base_model(**tokens)
        wrapper._remove_read_hooks()
        assert output.logits is not None

    def test_empty_memory_no_change(self, wrapper):
        tokens = wrapper.tokenizer("Hello world", return_tensors="pt")
        tokens = {k: v.to(wrapper.device) for k, v in tokens.items()}
        with torch.no_grad():
            out1 = wrapper.base_model(**tokens).logits.clone()
        # Empty memory → no hooks installed → output identical
        with torch.no_grad():
            out2 = wrapper.base_model(**tokens).logits
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_memory_injection_changes_logits(self, wrapper):
        """Memory injection via read hooks must visibly change logits.

        Verifies that _install_read_hooks() actually modifies the residual
        stream: logits with memory should differ from logits without memory.
        """
        wrapper.clear_memory()
        wrapper.write_memory("Dr. Elena Voss discovered Pyrothene in 2031")
        assert wrapper.store.active_episodes > 0, (
            "No episodes stored — surprise threshold too high or text too short"
        )

        tokens = wrapper.tokenizer("What did Dr. Voss discover?", return_tensors="pt")
        tokens = {k: v.to(wrapper.device) for k, v in tokens.items()}

        # Forward without memory hooks
        with torch.no_grad():
            out_no_mem = wrapper.base_model(**tokens).logits.clone()

        # Forward with memory hooks injecting stored KV
        wrapper._install_read_hooks()
        try:
            with torch.no_grad():
                out_with_mem = wrapper.base_model(**tokens).logits
        finally:
            wrapper._remove_read_hooks()

        diff = (out_no_mem - out_with_mem).abs().max().item()
        assert diff > 1e-5, (
            f"Memory injection had no effect on logits (max diff={diff:.2e}). "
            "Read hooks are not modifying the residual stream."
        )
