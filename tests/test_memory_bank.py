# tests/test_memory_bank.py
import torch
import pytest
import tempfile
import os
from nanochat.memory_bank import MemoryBank, MemoryBankConfig

CFG = MemoryBankConfig(d_key=16, d_value=8, max_slots=256)


class TestInit:
    def test_empty_bank(self):
        bank = MemoryBank(CFG)
        assert bank.size == 0

    def test_config_stored(self):
        bank = MemoryBank(CFG)
        assert bank.config.d_key == 16
        assert bank.config.d_value == 8
        assert bank.config.max_slots == 256


class TestWriteRead:
    def test_write_adds_slot(self):
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        value = torch.randn(CFG.d_value)
        bank.write(key, value, surprise=1.0)
        assert bank.size == 1

    def test_write_below_threshold_ignored(self):
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        value = torch.randn(CFG.d_value)
        bank.write(key, value, surprise=0.01)  # below default 0.5
        assert bank.size == 0

    def test_read_empty_returns_zeros(self):
        bank = MemoryBank(CFG)
        query = torch.randn(CFG.d_key)
        result = bank.read(query)
        assert result.shape == (CFG.d_value,)
        assert torch.allclose(result, torch.zeros(CFG.d_value))

    def test_read_retrieves_similar(self):
        torch.manual_seed(42)
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        key = key / key.norm()
        value = torch.ones(CFG.d_value)
        bank.write(key, value, surprise=1.0)

        # Query with same key should retrieve similar value
        result = bank.read(key)
        # With only one slot, softmax gives weight 1.0 to it
        assert torch.allclose(result, value, atol=1e-5)

    def test_read_prefers_similar_key(self):
        torch.manual_seed(42)
        bank = MemoryBank(CFG)
        # Write two slots: one close to query, one far
        key_close = torch.randn(CFG.d_key)
        key_close = key_close / key_close.norm()
        key_far = torch.randn(CFG.d_key)
        key_far = key_far / key_far.norm()
        val_close = torch.ones(CFG.d_value)
        val_far = -torch.ones(CFG.d_value)
        bank.write(key_close, val_close, surprise=1.0)
        bank.write(key_far, val_far, surprise=1.0)

        result = bank.read(key_close)
        # Result should be closer to val_close than val_far
        sim_close = torch.dot(result, val_close)
        sim_far = torch.dot(result, val_far)
        assert sim_close > sim_far, f"Should prefer similar: close={sim_close:.3f} far={sim_far:.3f}"


class TestDecay:
    def test_decay_reduces_usage(self):
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        value = torch.randn(CFG.d_value)
        bank.write(key, value, surprise=1.0)
        usage_before = bank._usage[0].item()
        bank.decay()
        usage_after = bank._usage[0].item()
        assert usage_after < usage_before

    def test_decay_increments_age(self):
        bank = MemoryBank(CFG)
        key = torch.randn(CFG.d_key)
        value = torch.randn(CFG.d_value)
        bank.write(key, value, surprise=1.0)
        bank.decay()
        assert bank._age[0].item() == 1

    def test_overwrite_low_usage(self):
        """When bank is full, new write should overwrite lowest-usage slot."""
        cfg = MemoryBankConfig(d_key=16, d_value=8, max_slots=3)
        bank = MemoryBank(cfg)
        # Fill bank
        for i in range(3):
            bank.write(torch.randn(cfg.d_key), torch.randn(cfg.d_value), surprise=1.0)
        assert bank.size == 3
        # Decay heavily so usage drops
        for _ in range(100):
            bank.decay()
        # Read slot 0 to boost its usage
        bank.read(bank._keys[0])
        # Now write a new slot -- should overwrite the lowest-usage one (not slot 0)
        new_value = torch.ones(cfg.d_value) * 99
        bank.write(torch.randn(cfg.d_key), new_value, surprise=1.0)
        assert bank.size == 3  # still 3, overwrote one
        # The new value should be in the bank
        assert (bank._values == 99).any()


class TestPersistence:
    def test_save_load_roundtrip(self):
        torch.manual_seed(42)
        bank = MemoryBank(CFG)
        for i in range(5):
            bank.write(torch.randn(CFG.d_key), torch.randn(CFG.d_value), surprise=1.0)
        bank.decay()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bank.pt")
            bank.save(path)
            bank2 = MemoryBank.load(path)

            assert bank2.size == bank.size
            assert torch.allclose(bank2._keys, bank._keys)
            assert torch.allclose(bank2._values, bank._values)
            assert torch.allclose(bank2._usage, bank._usage)
            assert torch.allclose(bank2._age, bank._age)

    def test_load_nonexistent_returns_empty_with_config(self):
        bank = MemoryBank.load("/nonexistent/path.pt", default_config=CFG)
        assert bank.size == 0
        assert bank.config.d_key == CFG.d_key

    def test_save_empty_bank(self):
        bank = MemoryBank(CFG)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bank.pt")
            bank.save(path)
            bank2 = MemoryBank.load(path)
            assert bank2.size == 0


class TestWorkingSet:
    def test_select_returns_k_slots(self):
        torch.manual_seed(42)
        cfg = MemoryBankConfig(d_key=16, d_value=8, max_slots=100, working_set_k=4)
        bank = MemoryBank(cfg)
        for i in range(20):
            bank.write(torch.randn(cfg.d_key), torch.randn(cfg.d_value), surprise=1.0)

        query = torch.randn(cfg.d_key)
        ws_keys, ws_values = bank.select_working_set(query)
        assert ws_keys.shape == (4, cfg.d_key)
        assert ws_values.shape == (4, cfg.d_value)

    def test_select_with_fewer_slots_than_k(self):
        cfg = MemoryBankConfig(d_key=16, d_value=8, max_slots=100, working_set_k=16)
        bank = MemoryBank(cfg)
        bank.write(torch.randn(cfg.d_key), torch.randn(cfg.d_value), surprise=1.0)
        query = torch.randn(cfg.d_key)
        ws_keys, ws_values = bank.select_working_set(query)
        assert ws_keys.shape[0] == 1  # only 1 slot exists

    def test_select_empty_bank(self):
        bank = MemoryBank(CFG)
        query = torch.randn(CFG.d_key)
        ws_keys, ws_values = bank.select_working_set(query)
        assert ws_keys.shape[0] == 0

    def test_select_returns_most_similar(self):
        torch.manual_seed(42)
        cfg = MemoryBankConfig(d_key=16, d_value=8, max_slots=100, working_set_k=1)
        bank = MemoryBank(cfg)
        # Write 10 random slots
        for i in range(10):
            bank.write(torch.randn(cfg.d_key), torch.randn(cfg.d_value), surprise=1.0)
        # Write one slot with known key
        target_key = torch.ones(cfg.d_key)
        target_value = torch.ones(cfg.d_value) * 42
        bank.write(target_key, target_value, surprise=1.0)
        # Query with the known key -- should retrieve it
        ws_keys, ws_values = bank.select_working_set(target_key)
        assert torch.allclose(ws_values[0], target_value, atol=1e-5)


class TestReadFromWorkingSet:
    def test_returns_correct_shape(self):
        bank = MemoryBank(CFG)
        ws_keys = torch.randn(4, CFG.d_key)
        ws_values = torch.randn(4, CFG.d_value)
        query = torch.randn(CFG.d_key)
        result = bank.read_from_working_set(query, ws_keys, ws_values)
        assert result.shape == (CFG.d_value,)

    def test_empty_working_set_returns_zeros(self):
        bank = MemoryBank(CFG)
        ws_keys = torch.zeros(0, CFG.d_key)
        ws_values = torch.zeros(0, CFG.d_value)
        query = torch.randn(CFG.d_key)
        result = bank.read_from_working_set(query, ws_keys, ws_values)
        assert torch.allclose(result, torch.zeros(CFG.d_value))

    def test_prefers_matching_key(self):
        torch.manual_seed(42)
        bank = MemoryBank(CFG)
        key_match = torch.randn(CFG.d_key); key_match = key_match / key_match.norm()
        key_other = torch.randn(CFG.d_key); key_other = key_other / key_other.norm()
        val_match = torch.ones(CFG.d_value)
        val_other = -torch.ones(CFG.d_value)
        ws_keys = torch.stack([key_match, key_other])
        ws_values = torch.stack([val_match, val_other])
        result = bank.read_from_working_set(key_match, ws_keys, ws_values)
        assert torch.dot(result, val_match) > torch.dot(result, val_other)
