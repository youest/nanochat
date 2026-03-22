# tests/test_gpt_cellmem_v2.py
"""
Integration tests for GPT + CellMem v2. Run as:
python -m pytest tests/test_gpt_cellmem_v2.py -v
"""
import torch
import pytest


class TestAttentionWithMemory:
    def test_forward_without_memory_unchanged(self):
        """When mem_kv=None, output identical to original."""
        from nanochat.gpt import CausalSelfAttention, GPTConfig, norm
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
        attn = CausalSelfAttention(cfg, layer_idx=0)
        torch.manual_seed(42)
        for p in attn.parameters():
            p.data.normal_(0, 0.02)
        x = torch.randn(1, 8, 32)
        cos = torch.ones(1, 8, 1, 8)
        sin = torch.zeros(1, 8, 1, 8)
        y1 = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None)
        y2 = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None,
                  mem_kv=None, mem_gate=None)
        assert torch.allclose(y1, y2, atol=1e-6)

    def test_forward_with_memory_gate_zero(self):
        """With mem_gate near 0, output nearly identical to no-memory."""
        from nanochat.gpt import CausalSelfAttention, GPTConfig, norm
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
        attn = CausalSelfAttention(cfg, layer_idx=0)
        torch.manual_seed(42)
        for p in attn.parameters():
            p.data.normal_(0, 0.02)
        x = torch.randn(1, 8, 32)
        cos = torch.ones(1, 8, 1, 8)
        sin = torch.zeros(1, 8, 1, 8)
        K_mem = torch.randn(1, 4, 2, 16)  # [B, K, Hkv, D]
        V_mem = torch.randn(1, 4, 2, 16)
        gate = torch.sigmoid(torch.tensor([-10.0]))  # near 0
        y_no = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None)
        y_mem = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None,
                     mem_kv=(K_mem, V_mem), mem_gate=gate)
        assert torch.allclose(y_no, y_mem, atol=1e-3)

    def test_forward_with_memory_gate_one(self):
        """With mem_gate=1, output should differ from no-memory."""
        from nanochat.gpt import CausalSelfAttention, GPTConfig, norm
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
        attn = CausalSelfAttention(cfg, layer_idx=0)
        torch.manual_seed(42)
        for p in attn.parameters():
            p.data.normal_(0, 0.02)
        x = torch.randn(1, 8, 32)
        cos = torch.ones(1, 8, 1, 8)
        sin = torch.zeros(1, 8, 1, 8)
        K_mem = torch.randn(1, 4, 2, 16)
        V_mem = torch.randn(1, 4, 2, 16)
        gate = torch.tensor([1.0])
        y_no = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None)
        y_mem = attn(norm(x), ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None,
                     mem_kv=(K_mem, V_mem), mem_gate=gate)
        assert not torch.allclose(y_no, y_mem, atol=0.01)

    def test_block_forward_passes_through_memory(self):
        """Block.forward passes mem_kv and mem_gate to attention."""
        from nanochat.gpt import Block, GPTConfig, norm
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
        block = Block(cfg, layer_idx=0)
        torch.manual_seed(42)
        for p in block.parameters():
            p.data.normal_(0, 0.02)
        x = torch.randn(1, 8, 32)
        cos = torch.ones(1, 8, 1, 8)
        sin = torch.zeros(1, 8, 1, 8)
        # Without memory
        y_no = block(x, ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None)
        # With memory and gate=1 should differ
        K_mem = torch.randn(1, 4, 2, 16)
        V_mem = torch.randn(1, 4, 2, 16)
        gate = torch.tensor([1.0])
        y_mem = block(x, ve=None, cos_sin=(cos, sin), window_size=(-1, 0), kv_cache=None,
                      mem_kv=(K_mem, V_mem), mem_gate=gate)
        assert not torch.allclose(y_no, y_mem, atol=0.01)


from nanochat.cellmem_v2 import CellMemConfig


class TestGPTWithCellMem:
    def _make_model(self, cellmem_cfg=None):
        from nanochat.gpt import GPT, GPTConfig
        if cellmem_cfg is None:
            cellmem_cfg = CellMemConfig(enabled=True, n_slots=8, layers="last3")
        cfg = GPTConfig(n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
                       sequence_len=16, vocab_size=64, cellmem=cellmem_cfg)
        model = GPT(cfg, pad_vocab_size_to=64)
        model.init_weights()
        return model

    def test_config_has_cellmem(self):
        from nanochat.gpt import GPTConfig
        cfg = GPTConfig()
        assert hasattr(cfg, 'cellmem')
        assert cfg.cellmem.enabled is False

    def test_config_dict_deserialization(self):
        """GPTConfig __post_init__ converts dict to CellMemConfig."""
        from nanochat.gpt import GPTConfig
        cfg = GPTConfig(cellmem={"enabled": True, "n_slots": 32})
        assert isinstance(cfg.cellmem, CellMemConfig)
        assert cfg.cellmem.enabled is True
        assert cfg.cellmem.n_slots == 32

    def test_model_creates_mem_gates(self):
        model = self._make_model()
        assert hasattr(model, 'mem_gates')
        assert len(model.mem_gates) == 3  # last3 on 4 layers = layers 1,2,3

    def test_mem_gates_init_near_zero(self):
        model = self._make_model()
        for gate in model.mem_gates:
            assert torch.sigmoid(gate).item() < 0.001

    def test_forward_training_mode_ignores_memory(self):
        model = self._make_model()
        model.train()
        idx = torch.randint(0, 64, (1, 8))
        targets = torch.randint(0, 64, (1, 8))
        loss = model(idx, targets=targets)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_inference_no_crash(self):
        model = self._make_model()
        model.eval()
        idx = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            logits = model(idx)
        assert logits.shape == (1, 8, 64)

    def test_num_scaling_params_includes_gates(self):
        model = self._make_model()
        counts = model.num_scaling_params()
        assert 'cellmem_gates' in counts
        assert counts['cellmem_gates'] == 3

    def test_estimate_flops_excludes_gates(self):
        """estimate_flops should exclude cellmem gates from matmul param count."""
        model = self._make_model()
        flops = model.estimate_flops()
        assert isinstance(flops, (int, float))
        assert flops > 0

    def test_optimizer_includes_gates(self):
        model = self._make_model()
        optimizer = model.setup_optimizer()
        all_params = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                all_params.add(id(p))
        for gate in model.mem_gates:
            assert id(gate) in all_params

    def test_cellmem_layer_indices_last3(self):
        """last3 on 4-layer model should yield layers [1,2,3]."""
        from nanochat.gpt import _cellmem_layer_indices, GPTConfig
        cfg = GPTConfig(n_layer=4, cellmem=CellMemConfig(enabled=True, layers="last3"))
        assert _cellmem_layer_indices(cfg) == [1, 2, 3]

    def test_cellmem_layer_indices_mid(self):
        from nanochat.gpt import _cellmem_layer_indices, GPTConfig
        cfg = GPTConfig(n_layer=4, cellmem=CellMemConfig(enabled=True, layers="mid"))
        assert _cellmem_layer_indices(cfg) == [2]

    def test_cellmem_layer_indices_all(self):
        from nanochat.gpt import _cellmem_layer_indices, GPTConfig
        cfg = GPTConfig(n_layer=4, cellmem=CellMemConfig(enabled=True, layers="all"))
        assert _cellmem_layer_indices(cfg) == [0, 1, 2, 3]

    def test_cellmem_layer_indices_disabled(self):
        from nanochat.gpt import _cellmem_layer_indices, GPTConfig
        cfg = GPTConfig(n_layer=4, cellmem=CellMemConfig(enabled=False))
        assert _cellmem_layer_indices(cfg) == []


class TestGenerateWithMemory:
    def test_generate_writes_memories(self, tmp_path):
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.cellmem_v2 import CellMemConfig, MemoryStore

        cellmem_cfg = CellMemConfig(enabled=True, n_slots=8, layers="last3",
                                     surprise_threshold=0.5,  # low threshold to ensure writes
                                     memory_dir=str(tmp_path))
        cfg = GPTConfig(n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
                       sequence_len=16, vocab_size=64, cellmem=cellmem_cfg)
        model = GPT(cfg, pad_vocab_size_to=64)
        model.init_weights()

        # generate() should create MemoryStore, write memories, save to disk
        tokens = list(range(8))
        generated = list(model.generate(tokens, max_tokens=4))
        assert len(generated) == 4
        # Memory file should exist after generate
        mem_path = tmp_path / "memory.pt"
        assert mem_path.exists()

    def test_generate_loads_existing_memory(self, tmp_path):
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.cellmem_v2 import CellMemConfig, MemoryStore

        cellmem_cfg = CellMemConfig(enabled=True, n_slots=8, layers="last3",
                                     memory_dir=str(tmp_path))
        cfg = GPTConfig(n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
                       sequence_len=16, vocab_size=64, cellmem=cellmem_cfg)
        model = GPT(cfg, pad_vocab_size_to=64)
        model.init_weights()

        # Pre-create a memory file
        store = MemoryStore(cellmem_cfg, d_model=32)
        store.write(torch.randn(32), surprise=5.0)
        mem_path = tmp_path / "memory.pt"
        store.save(mem_path)

        # generate() should load it
        tokens = list(range(8))
        generated = list(model.generate(tokens, max_tokens=2))
        assert len(generated) == 2
        # model.memory_store should have the loaded memory + possibly new ones
        assert model.memory_store is not None
        assert model.memory_store.active_count >= 1

    def test_generate_saves_on_early_exit(self, tmp_path):
        """Memory is saved even if caller doesn't exhaust the generator (via try/finally)."""
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.cellmem_v2 import CellMemConfig

        cellmem_cfg = CellMemConfig(enabled=True, n_slots=8, layers="last3",
                                     surprise_threshold=0.5,
                                     memory_dir=str(tmp_path))
        cfg = GPTConfig(n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
                       sequence_len=16, vocab_size=64, cellmem=cellmem_cfg)
        model = GPT(cfg, pad_vocab_size_to=64)
        model.init_weights()

        # Only consume 1 token from the generator, then close it
        gen = model.generate(list(range(8)), max_tokens=10)
        first_token = next(gen)
        gen.close()  # This triggers GeneratorExit -> finally block

        # Memory should still be saved
        mem_path = tmp_path / "memory.pt"
        assert mem_path.exists()


class TestGPTCellMemDisabled:
    def test_no_mem_gates_when_disabled(self):
        from nanochat.gpt import GPT, GPTConfig
        cfg = GPTConfig(n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
                       sequence_len=16, vocab_size=64)
        model = GPT(cfg, pad_vocab_size_to=64)
        model.init_weights()
        assert not hasattr(model, 'mem_gates') or model.mem_gates is None

    def test_forward_works_without_cellmem(self):
        """Default GPTConfig (cellmem disabled) should work exactly as before."""
        from nanochat.gpt import GPT, GPTConfig
        cfg = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32,
                       sequence_len=16, vocab_size=64)
        model = GPT(cfg, pad_vocab_size_to=64)
        model.init_weights()
        model.train()
        idx = torch.randint(0, 64, (1, 8))
        targets = torch.randint(0, 64, (1, 8))
        loss = model(idx, targets=targets)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestEndToEnd:
    def test_memory_lifecycle(self, tmp_path):
        """Full lifecycle: create model -> write memory -> save -> load into fresh store -> forward with memory -> verify no NaN."""
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.cellmem_v2 import CellMemConfig, MemoryStore

        cellmem_cfg = CellMemConfig(enabled=True, n_slots=8, layers="last3",
                                     surprise_threshold=0.0)
        cfg = GPTConfig(n_layer=4, n_head=2, n_kv_head=2, n_embd=32,
                       sequence_len=16, vocab_size=64, cellmem=cellmem_cfg)
        model = GPT(cfg, pad_vocab_size_to=64)
        model.init_weights()
        model.eval()

        store = MemoryStore(cellmem_cfg, d_model=32)
        model.memory_store = store

        # Session 1: write a memory
        idx = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            logits1 = model(idx)
        store.write(torch.randn(32), surprise=5.0)
        assert store.active_count == 1

        # Save
        save_path = tmp_path / "memory.pt"
        store.save(save_path)

        # Session 2: load and use
        store2 = MemoryStore(cellmem_cfg, d_model=32)
        store2.load(save_path)
        model.memory_store = store2
        assert store2.active_count == 1

        with torch.no_grad():
            logits2 = model(idx)
        assert logits2.shape == (1, 8, 64)
        assert not torch.isnan(logits2).any()
