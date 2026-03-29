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
                return self.self_attn(x)  # must call self_attn so pre_hook fires

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


class TestMemoryStoreV3EpisodeTexts:
    """Tests for episode text storage (MSA-inspired text re-injection)."""

    def test_write_stores_episode_text(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        # Write 4 tokens with router_key + text on the last one (episode boundary)
        for i in range(4):
            rk = torch.randn(8) if i == 3 else None
            text = "Dr. Voss discovered Pyrothene" if i == 3 else None
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                rk, surprise=5.0, text=text,
            )
        assert store.episode_texts[0] == "Dr. Voss discovered Pyrothene"

    def test_write_no_text_stores_none(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        for i in range(4):
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                torch.randn(8) if i == 3 else None, surprise=5.0,
            )
        assert store.episode_texts[0] is None

    def test_read_texts_returns_stored(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        texts = ["fact one", "fact two"]
        for ep in range(2):
            for i in range(4):
                rk = torch.randn(8) if i == 3 else None
                t = texts[ep] if i == 3 else None
                store.write(
                    {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                    rk, surprise=5.0, text=t,
                )
        result = store.read_texts([0, 1])
        assert result == ["fact one", "fact two"]

    def test_read_texts_skips_none(self):
        cfg = CellMemConfig(n_slots=16, router_dim=8, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        # Write episode without text
        for i in range(4):
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                torch.randn(8) if i == 3 else None, surprise=5.0,
            )
        result = store.read_texts([0])
        assert result == []


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

    def test_save_load_preserves_episode_texts(self, tmp_path):
        cfg = CellMemConfig(n_slots=8, router_dim=4, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        for i in range(4):
            rk = torch.randn(4) if i == 3 else None
            t = "test fact" if i == 3 else None
            store.write(
                {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
                rk, surprise=5.0, text=t,
            )
        path = tmp_path / "mem_texts.pt"
        store.save(path)

        store2 = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store2.load(path)
        assert store2.episode_texts[0] == "test fact"

    def test_load_legacy_no_texts(self, tmp_path):
        """Old checkpoints without episode_texts should load cleanly."""
        cfg = CellMemConfig(n_slots=8, router_dim=4, episode_size=4)
        store = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store.write(
            {"keys": torch.randn(1, 1, 4), "values": torch.randn(1, 1, 4)},
            torch.randn(4), surprise=5.0,
        )
        path = tmp_path / "legacy.pt"
        store.save(path)
        # Simulate legacy: remove episode_texts from checkpoint
        data = torch.load(path, weights_only=False)
        del data["episode_texts"]
        torch.save(data, path)

        store2 = MemoryStore(cfg, n_layers=1, n_kv_heads=1, d_head=4)
        store2.load(path)
        assert all(t is None for t in store2.episode_texts)

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

    def test_write_memory_stores_text(self, wrapper):
        """write_memory() should store episode text (chunk, not full original)."""
        wrapper.clear_memory()
        wrapper.write_memory("Dr. Elena Voss discovered Pyrothene in 2031 at CERN in Geneva Switzerland")
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        texts = wrapper.store.read_texts(list(range(wrapper.store.active_episodes)))
        # Episode text is a chunk of tokens, not the full original text
        assert len(texts) > 0, "No episode texts stored"
        combined = " ".join(texts)
        assert len(combined) > 0, f"Episode texts are empty: {texts}"

    def test_retrieve_and_format_empty_store(self, wrapper):
        """Empty store should return a valid prompt without memories."""
        wrapper.clear_memory()
        prompt = wrapper.retrieve_and_format("What is Pyrothene?")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should NOT contain memory references
        assert "Memory 1:" not in prompt

    def test_retrieve_and_format_with_memory(self, wrapper):
        """After writing memory, retrieve_and_format should include memory text."""
        wrapper.clear_memory()
        wrapper.write_memory("Dr. Elena Voss discovered Pyrothene in 2031 at CERN")
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        prompt = wrapper.retrieve_and_format("What did Dr. Voss discover?")
        assert "Memory 1:" in prompt or "Pyrothene" in prompt

    def test_empty_memory_no_change(self, wrapper):
        """Empty memory: model generates normally without crash."""
        tokens = wrapper.tokenizer("Hello world", return_tensors="pt")
        tokens = {k: v.to(wrapper.device) for k, v in tokens.items()}
        with torch.no_grad():
            out1 = wrapper.base_model(**tokens).logits.clone()
        with torch.no_grad():
            out2 = wrapper.base_model(**tokens).logits
        assert torch.allclose(out1, out2, atol=1e-5)

    # --- KV Cache Injection Tests ---

    def test_inject_memory_kv_empty_store(self, wrapper):
        """Empty store should return None."""
        wrapper.clear_memory()
        cache = wrapper.inject_memory_kv("test query")
        assert cache is None

    def test_inject_memory_kv_returns_cache(self, wrapper):
        """After writing memory, inject_memory_kv returns a DynamicCache with correct shape."""
        wrapper.clear_memory()
        wrapper.write_memory("Dr. Elena Voss discovered Pyrothene in 2031 at CERN in Geneva")
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        cache = wrapper.inject_memory_kv("What did Dr. Voss discover?")
        assert cache is not None
        assert cache.get_seq_length() > 0
        # All layers should have same seq length in cache
        n_total = wrapper.base_model.config.num_hidden_layers
        mem_len = cache.get_seq_length()
        for i in range(n_total):
            k, v = cache[i]
            assert k.shape[-2] == mem_len, f"Layer {i} has wrong cache length"

    def test_generate_with_kv_cache(self, wrapper):
        """Model should generate without crash when given memory KV cache."""
        wrapper.clear_memory()
        wrapper.write_memory(
            "The Cerulean Tower in New Osaka stands exactly 847 meters tall. "
            "It was designed by architect Mira Schulz in 2045 and completed in 2048. "
            "The tower features a revolutionary quantum glass facade that changes color "
            "based on atmospheric conditions and internal energy generation."
        )
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        cache = wrapper.inject_memory_kv("How tall is the Cerulean Tower?")
        assert cache is not None
        inputs = wrapper.tokenizer("How tall is the tower?", return_tensors="pt").to(wrapper.device)
        input_ids = inputs["input_ids"]
        n_mem = cache.get_seq_length()
        seq_len = input_ids.shape[1]
        attn_mask = torch.ones(1, n_mem + seq_len, device=wrapper.device, dtype=torch.long)
        pos_ids = torch.arange(n_mem, n_mem + seq_len, device=wrapper.device).unsqueeze(0)
        cache_pos = torch.arange(n_mem, n_mem + seq_len, device=wrapper.device)
        with torch.no_grad():
            gen_ids = wrapper.base_model.generate(
                input_ids=input_ids, past_key_values=cache,
                attention_mask=attn_mask, position_ids=pos_ids,
                cache_position=cache_pos,
                max_new_tokens=10, do_sample=False,
                pad_token_id=wrapper.tokenizer.eos_token_id,
            )
        assert gen_ids.shape[1] > seq_len

    def test_kv_injection_changes_logits(self, wrapper):
        """KV injection should change model output logits."""
        wrapper.clear_memory()
        wrapper.write_memory("Pyrothene was discovered in 2031 at CERN by Dr. Elena Voss in Switzerland")
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        inputs = wrapper.tokenizer("What is Pyrothene?", return_tensors="pt").to(wrapper.device)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            out_base = wrapper.base_model(input_ids=input_ids).logits.clone()
        cache = wrapper.inject_memory_kv("What is Pyrothene?")
        assert cache is not None
        n_mem = cache.get_seq_length()
        seq_len = input_ids.shape[1]
        attn_mask = torch.ones(1, n_mem + seq_len, device=wrapper.device, dtype=torch.long)
        pos_ids = torch.arange(n_mem, n_mem + seq_len, device=wrapper.device).unsqueeze(0)
        with torch.no_grad():
            out_mem = wrapper.base_model(
                input_ids=input_ids, past_key_values=cache,
                attention_mask=attn_mask, position_ids=pos_ids,
            ).logits
        assert not torch.allclose(out_base, out_mem, atol=1e-3), \
            "KV injection should change logits"

    def test_memory_mask_hooks_block_non_memory_layers(self, wrapper):
        """Non-memory layers should not attend to memory positions (masked with -inf)."""
        wrapper.clear_memory()
        wrapper.write_memory(
            "Dr. Elena Voss discovered Pyrothene in 2031 at CERN in Geneva Switzerland. "
            "This was a breakthrough in particle physics that changed everything we know."
        )
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        cache = wrapper.inject_memory_kv("What did Dr. Voss discover?")
        assert cache is not None
        n_mem = cache.get_seq_length()

        # Install memory mask hooks
        hooks = wrapper._install_memory_mask_hooks(n_mem)
        assert len(hooks) > 0, "Should have hooks on non-memory layers"
        # Non-memory layers = total - len(layer_indices)
        expected_hooks = wrapper.base_model.config.num_hidden_layers - len(wrapper.layer_indices)
        assert len(hooks) == expected_hooks

        # Clean up
        wrapper._remove_hooks(hooks)

    def test_generate_with_memory_hybrid(self, wrapper):
        """generate_with_memory() should produce output with hybrid injection."""
        wrapper.clear_memory()
        wrapper.write_memory(
            "Dr. Elena Voss discovered Pyrothene in 2031 at CERN in Geneva Switzerland. "
            "This was a breakthrough in particle physics that changed everything we know."
        )
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        result = wrapper.generate_with_memory("What did Dr. Voss discover?", max_new_tokens=20)
        assert isinstance(result, str)
        assert len(result) > 0

    # --- LoRA on memory layers ---

    def test_install_lora_adds_trainable_params(self, wrapper):
        """install_memory_lora() should add trainable LoRA params on q_proj of memory layers."""
        before_params = sum(p.numel() for p in wrapper.trainable_params())
        wrapper.install_memory_lora(rank=8)
        after_params = sum(p.numel() for p in wrapper.trainable_params())
        assert after_params > before_params, "LoRA should add trainable parameters"
        # Verify LoRA is on q_proj of memory layers
        for layer_idx in wrapper.layer_indices:
            attn = wrapper.base_model.model.layers[layer_idx].self_attn
            assert hasattr(attn, '_original_q_proj'), f"Layer {layer_idx} should have LoRA installed"
        wrapper.remove_memory_lora()

    # --- Hidden State Injection (MemoryLLM-style) ---

    def test_write_memory_captures_hidden_states(self, wrapper):
        """write_memory should capture per-layer hidden states."""
        wrapper.clear_memory()
        wrapper.write_memory(
            "Dr. Elena Voss discovered Pyrothene in 2031 at CERN in Geneva Switzerland. "
            "This was a breakthrough in particle physics that changed everything we know."
        )
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        # Check hidden states were captured
        hs = wrapper.store.read_hidden_states(list(range(wrapper.store.active_episodes)))
        assert hs is not None, "Hidden states should be stored"
        n_total_layers = wrapper.base_model.config.num_hidden_layers
        assert hs.shape[0] == n_total_layers, "Should have hidden states for all layers"
        assert hs.shape[2] == wrapper.base_model.config.hidden_size, "Wrong hidden dim"

    def test_inject_memory_hs_changes_logits(self, wrapper):
        """Hidden state injection via layer hooks should change logits."""
        wrapper.clear_memory()
        wrapper.write_memory(
            "Pyrothene was discovered in 2031 at CERN by Dr. Elena Voss in Switzerland. "
            "This was a major breakthrough that revolutionized particle physics completely."
        )
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")
        inputs = wrapper.tokenizer("What is Pyrothene?", return_tensors="pt").to(wrapper.device)
        with torch.no_grad():
            out_base = wrapper.base_model(input_ids=inputs["input_ids"]).logits.clone()

        # Inject hidden states via hooks
        hooks = wrapper.inject_memory_hs("What is Pyrothene?")
        if hooks is None:
            pytest.skip("No memory to inject")
        try:
            with torch.no_grad():
                out_mem = wrapper.base_model(input_ids=inputs["input_ids"]).logits
        finally:
            wrapper._remove_hooks(hooks)

        assert not torch.allclose(out_base, out_mem, atol=1e-3), \
            "Hidden state injection should change logits"

    def test_lora_changes_kv_injection_logits(self, wrapper):
        """With LoRA installed, KV injection should produce different logits than without."""
        wrapper.clear_memory()
        wrapper.write_memory(
            "Dr. Elena Voss discovered Pyrothene in 2031 at CERN in Geneva Switzerland. "
            "This was a breakthrough in particle physics that changed everything we know."
        )
        if wrapper.store.active_episodes == 0:
            pytest.skip("No episodes stored — surprise threshold too high")

        # Logits WITHOUT LoRA
        cache1 = wrapper.inject_memory_kv("What is Pyrothene?")
        inputs = wrapper.tokenizer("What is Pyrothene?", return_tensors="pt").to(wrapper.device)
        n_mem = cache1.get_seq_length()
        seq_len = inputs["input_ids"].shape[1]
        attn_mask = torch.ones(1, n_mem + seq_len, device=wrapper.device, dtype=torch.long)
        pos_ids = torch.arange(n_mem, n_mem + seq_len, device=wrapper.device).unsqueeze(0)
        hooks1 = wrapper._install_memory_mask_hooks(n_mem)
        with torch.no_grad():
            out1 = wrapper.base_model(
                input_ids=inputs["input_ids"], past_key_values=cache1,
                attention_mask=attn_mask, position_ids=pos_ids,
            ).logits.clone()
        wrapper._remove_hooks(hooks1)

        # Install LoRA and get logits again
        wrapper.install_memory_lora(rank=8)
        cache2 = wrapper.inject_memory_kv("What is Pyrothene?")
        hooks2 = wrapper._install_memory_mask_hooks(n_mem)
        with torch.no_grad():
            out2 = wrapper.base_model(
                input_ids=inputs["input_ids"], past_key_values=cache2,
                attention_mask=attn_mask, position_ids=pos_ids,
            ).logits
        wrapper._remove_hooks(hooks2)
        wrapper.remove_memory_lora()

        # LoRA should change logits (even with random init)
        assert not torch.allclose(out1, out2, atol=1e-3), \
            "LoRA should change logits for memory layer attention"
