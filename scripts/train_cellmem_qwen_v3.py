# scripts/train_cellmem_qwen_v3.py
"""
CellMem v3 Training Script.

Two-phase training: router warmup (InfoNCE) then generation validation.
Only router (W_Q^R, W_K^R) is trained. Backbone always frozen.

Usage:
    python scripts/train_cellmem_qwen_v3.py --model Qwen/Qwen3-4B --epochs1 20 --epochs2 30
"""
from __future__ import annotations
import argparse
import contextlib
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanochat.cellmem_v3 import CellMemConfig, MemoryRouter, MemoryStore
from nanochat.cellmem_v2 import SurpriseCalculator
from nanochat.kv_interceptor import KVInterceptor


# ---------------------------------------------------------------------------
# LoRA adapter for memory-layer attention
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter: output = base(x) + scale * B @ A @ x"""
    def __init__(self, base: nn.Linear, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = scale
        # Init: A normal, B small random (NOT zero — zero blocks gradient to A)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.normal_(self.lora_B.weight, std=1e-3)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

FICTIONAL_FACTS = [
    ("Dr. Elena Voss discovered Pyrothene in 2031 at CERN", "What did Dr. Voss discover?", "Pyrothene"),
    ("The Verdant Protocol was signed on March 14, 2045 in Geneva", "When was the Verdant Protocol signed?", "March 14, 2045"),
    ("Nexon-7 is a moon of the exoplanet Korrath discovered in 2038", "What is Nexon-7?", "a moon of Korrath"),
    ("Dr. Raj Patel invented the Quantum Lattice Compiler in 2029", "Who invented the Quantum Lattice Compiler?", "Dr. Raj Patel"),
    ("The Cerulean Tower in New Osaka stands 847 meters tall", "How tall is the Cerulean Tower?", "847 meters"),
    ("Project Helios achieved nuclear fusion ignition in 2034 at MIT", "What did Project Helios achieve?", "nuclear fusion ignition"),
    ("The Amber Sea was formed by the Ryazhan meteor impact in 2027", "How was the Amber Sea formed?", "Ryazhan meteor impact"),
    ("Dr. Linh Nguyen developed the BioMesh neural interface in 2041", "Who developed the BioMesh?", "Dr. Linh Nguyen"),
    ("The Solaris Agreement banned orbital weapons in 2037", "What did the Solaris Agreement ban?", "orbital weapons"),
    ("Professor Kenji Tanaka discovered the Tanaka Constant in 2033", "What did Professor Tanaka discover?", "the Tanaka Constant"),
    ("The Helix Bridge in Porto Alegre spans 2.3 kilometers", "How long is the Helix Bridge?", "2.3 kilometers"),
    ("Zarya Station became humanity's first Mars colony in 2052", "What was humanity's first Mars colony?", "Zarya Station"),
    ("The Omnisense chip processes 1 petaflop per milliwatt since 2040", "What does the Omnisense chip achieve?", "1 petaflop per milliwatt"),
    ("Dr. Amara Osei won the Nobel Prize for synthetic photosynthesis in 2036", "Who won the Nobel for synthetic photosynthesis?", "Dr. Amara Osei"),
    ("The Cobalt Virus pandemic of 2030 affected 2.1 billion people", "How many people did the Cobalt Virus affect?", "2.1 billion"),
    ("Architect Mira Schulz designed the Crystal Parliament in Berlin", "Who designed the Crystal Parliament?", "Mira Schulz"),
]

# Conversational memory pairs — names, jobs, preferences, personal info
CONVERSATIONAL_MEMORY = [
    # Names
    ("User: Hello! My name is Jack.\nAssistant: Nice to meet you, Jack!", "What is my name?", "Your name is Jack."),
    ("User: Hi, I'm Sarah.\nAssistant: Hi Sarah, nice to meet you!", "What's my name?", "Your name is Sarah."),
    ("User: My name is Marco, nice to meet you.\nAssistant: Ciao Marco!", "Who am I?", "You are Marco."),
    ("User: I'm Giuseppe.\nAssistant: Piacere Giuseppe!", "What is my name?", "Your name is Giuseppe."),
    ("User: Call me Elena.\nAssistant: Sure, Elena!", "What should I call you?", "I should call you Elena."),
    ("User: I'm Alex, I just joined the team.\nAssistant: Welcome Alex!", "What's my name?", "Your name is Alex."),
    ("User: Hey, it's David here.\nAssistant: Hi David!", "Who am I?", "You are David."),
    ("User: My name is Yuki, I'm from Tokyo.\nAssistant: Nice to meet you, Yuki!", "What is my name?", "Your name is Yuki."),
    # Jobs and work
    ("User: I work as a CTO at a startup called Fairmind.\nAssistant: That's exciting!", "Where do I work?", "You work at a startup called Fairmind."),
    ("User: I'm a data scientist at Google.\nAssistant: Interesting role!", "What do I do for work?", "You are a data scientist at Google."),
    ("User: I'm a teacher at a high school in Rome.\nAssistant: That's wonderful!", "What is my job?", "You are a teacher at a high school in Rome."),
    ("User: I run a bakery in Paris.\nAssistant: How lovely!", "What do I do?", "You run a bakery in Paris."),
    ("User: I'm an AI researcher at DeepMind.\nAssistant: Fascinating work!", "Where do I work?", "You work at DeepMind as an AI researcher."),
    ("User: I'm a doctor, I work at the city hospital.\nAssistant: Important work!", "What is my profession?", "You are a doctor at the city hospital."),
    ("User: I'm a freelance designer.\nAssistant: Creative work!", "What do I do for a living?", "You are a freelance designer."),
    ("User: I work on LLM architectures and agent systems.\nAssistant: Cutting edge!", "What do I work on?", "You work on LLM architectures and agent systems."),
    # Personal info
    ("User: I was born in Milan in 1990.\nAssistant: Beautiful city!", "Where was I born?", "You were born in Milan."),
    ("User: I'm 35 years old.\nAssistant: Got it!", "How old am I?", "You are 35 years old."),
    ("User: I have two cats named Luna and Stella.\nAssistant: Cute names!", "What are my cats' names?", "Your cats are named Luna and Stella."),
    ("User: My favorite color is blue.\nAssistant: Nice choice!", "What's my favorite color?", "Your favorite color is blue."),
    ("User: I live in Berlin.\nAssistant: Great city!", "Where do I live?", "You live in Berlin."),
    ("User: I'm allergic to peanuts.\nAssistant: I'll remember that.", "What am I allergic to?", "You are allergic to peanuts."),
    ("User: My birthday is on December 15th.\nAssistant: Noted!", "When is my birthday?", "Your birthday is on December 15th."),
    ("User: I speak Italian and English fluently.\nAssistant: Bilingual!", "What languages do I speak?", "You speak Italian and English."),
    # Preferences
    ("User: I prefer Python over JavaScript.\nAssistant: Good choice!", "What programming language do I prefer?", "You prefer Python."),
    ("User: I love pizza margherita.\nAssistant: Classic!", "What's my favorite food?", "You love pizza margherita."),
    ("User: I'm a morning person, I wake up at 6am.\nAssistant: Early riser!", "When do I usually wake up?", "You wake up at 6am."),
    ("User: I'm reading a book called Dune right now.\nAssistant: Great book!", "What book am I reading?", "You are reading Dune."),
    # Multi-turn context
    ("User: I just got back from a trip to Japan.\nAssistant: How was it?", "Where did I travel recently?", "You recently traveled to Japan."),
    ("User: I'm working on a project called CellMem.\nAssistant: Tell me more!", "What project am I working on?", "You are working on a project called CellMem."),
    ("User: My team has 5 people.\nAssistant: A compact team!", "How many people are on my team?", "Your team has 5 people."),
    ("User: We just raised 2 million in funding.\nAssistant: Congratulations!", "How much funding did we raise?", "You raised 2 million in funding."),
]


def generate_training_data(n: int = 100) -> list[dict]:
    data = []
    facts = FICTIONAL_FACTS * (n // len(FICTIONAL_FACTS) + 1)
    random.shuffle(facts)
    for memory, query, answer in facts[:n]:
        data.append({"memory": memory, "query": query, "answer": answer})
    return data


def generate_conversational_data(n: int = 200) -> list[dict]:
    """Generate training data mixing conversational memory + fictional facts."""
    data = []
    # Conversational pairs
    conv = CONVERSATIONAL_MEMORY * (n // len(CONVERSATIONAL_MEMORY) + 1)
    random.shuffle(conv)
    for memory, query, answer in conv[:n // 2]:
        data.append({"memory": memory, "query": query, "answer": answer})
    # Fictional facts (structured)
    facts = FICTIONAL_FACTS * (n // len(FICTIONAL_FACTS) + 1)
    random.shuffle(facts)
    for memory, query, answer in facts[:n // 2]:
        data.append({"memory": memory, "query": query, "answer": answer})
    random.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# CellMemWrapper
# ---------------------------------------------------------------------------

class CellMemWrapper:
    """Orchestrates KVInterceptor + MemoryRouter + MemoryStore for Qwen."""

    def __init__(self, model, tokenizer, layer_indices: list[int],
                 config: CellMemConfig | None = None, device: str = "cpu"):
        self.base_model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer_indices = layer_indices

        d_model = model.config.hidden_size
        n_kv_heads = model.config.num_key_value_heads
        # Use actual k_proj output to get d_head (Qwen3 has head_dim != hidden_size/n_heads)
        k_proj = model.model.layers[0].self_attn.k_proj
        d_head = k_proj.out_features // n_kv_heads
        self._d_head = d_head  # store for clear_memory()

        self.config = config or CellMemConfig(
            router_layers=layer_indices,
            episode_size=8,
            top_k=4,
        )

        self.router = MemoryRouter(
            d_model=d_model,
            d_router=self.config.router_dim,
            top_k=self.config.top_k,
        ).to(device)

        self.store = MemoryStore(
            self.config,
            n_layers=len(layer_indices),
            n_kv_heads=n_kv_heads,
            d_head=d_head,
        )

        self.surprise_calc = SurpriseCalculator(self.config)
        # Intercept ALL layers: K/V for router layers, hidden states for all
        all_layers = list(range(model.config.num_hidden_layers))
        self.interceptor = KVInterceptor(
            model, all_layers, model_type="qwen", capture_hidden_states=True)
        self._episode_buffer: list = []  # hidden states for current episode
        self._episode_token_count = 0
        self._episode_token_ids: list = []  # token ids for current episode (for text decoding)
        self._current_write_text: str | None = None

        # Freeze backbone
        for p in model.parameters():
            p.requires_grad_(False)

    def trainable_params(self):
        params = list(self.router.parameters())
        # Add LoRA params if installed (any layer)
        for layer_idx in range(self.base_model.config.num_hidden_layers):
            attn = self.base_model.model.layers[layer_idx].self_attn
            if hasattr(attn, '_original_q_proj'):
                lora = attn.q_proj
                params.extend(lora.lora_A.parameters())
                params.extend(lora.lora_B.parameters())
        return params

    def install_memory_lora(self, rank: int = 8, scale: float = 1.0,
                              all_layers: bool = False):
        """Install LoRA on q_proj to learn to attend to memory.

        Args:
            all_layers: If True, install on ALL layers (for HS injection).
                       If False, only on memory layers (for KV injection).
        """
        target_layers = (list(range(self.base_model.config.num_hidden_layers))
                        if all_layers else self.layer_indices)
        for layer_idx in target_layers:
            attn = self.base_model.model.layers[layer_idx].self_attn
            if hasattr(attn, '_original_q_proj'):
                continue  # Already installed
            original = attn.q_proj
            dtype = next(self.base_model.parameters()).dtype
            lora = LoRALinear(original, rank=rank, scale=scale).to(device=self.device, dtype=dtype)
            attn._original_q_proj = original
            attn.q_proj = lora

    def remove_memory_lora(self):
        """Remove LoRA from all layers that have it, restoring original q_proj."""
        for layer_idx in range(self.base_model.config.num_hidden_layers):
            attn = self.base_model.model.layers[layer_idx].self_attn
            if hasattr(attn, '_original_q_proj'):
                attn.q_proj = attn._original_q_proj
                del attn._original_q_proj

    def _is_text_redundant(self, text: str, threshold: float = 0.7) -> bool:
        """Check if text is too similar to an already stored episode text."""
        existing = self.store.read_texts(list(range(self.store.active_episodes)))
        if not existing:
            return False
        # Simple word-overlap check (Jaccard on words)
        new_words = set(text.lower().split())
        for stored in existing:
            stored_words = set(stored.lower().split())
            if not new_words or not stored_words:
                continue
            overlap = len(new_words & stored_words)
            union = len(new_words | stored_words)
            if union > 0 and overlap / union > threshold:
                return True
        return False

    def write_memory(self, text: str):
        """Process text and write surprising tokens to memory store."""
        if self._is_text_redundant(text):
            return
        self._current_write_text = text
        self._text_used = False  # only attach text to the first episode
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # [1, T]
        T = input_ids.shape[1]
        if T < 2:
            return

        self.interceptor.clear_buffer()

        with torch.no_grad():
            outputs = self.base_model(**inputs, output_hidden_states=True)

        logits = outputs.logits  # [1, T, V]
        targets = torch.cat([input_ids[:, 1:], input_ids[:, -1:]], dim=1)  # shift
        surprises = self.surprise_calc.compute_surprise(logits, targets)  # [1, T]
        write_mask = self.surprise_calc.get_write_mask(surprises)  # [1, T] bool

        kv_buf = self.interceptor.get_buffered_kv()  # {layer_idx: (K, V)}
        hs_buf = self.interceptor.get_buffered_hidden_states()  # {layer_idx: [1, T, D]}
        if not kv_buf:
            return

        # hidden_states is a tuple of (n_hidden_layers+1) tensors; index 0 = embedding output,
        # index i = output of layer i-1. Use layer index + 1 to get post-layer hidden states.
        hidden_states = outputs.hidden_states  # tuple of [1, T, D] per layer (+1 for embedding)
        first_hs_idx = min(self.layer_indices[0] + 1, len(hidden_states) - 1)
        first_router_layer_hs = hidden_states[first_hs_idx]  # [1, T, D]

        # Track which token indices are written (for hidden state slicing)
        written_token_indices = []

        for t in range(T):
            if not write_mask[0, t].item():
                continue

            written_token_indices.append(t)

            # Build per-layer K/V dict for this token
            kv_list_k = []
            kv_list_v = []
            for li, layer_idx in enumerate(self.layer_indices):
                if layer_idx in kv_buf:
                    k, v = kv_buf[layer_idx]  # [1, T, n_kv_heads * d_head]
                    k_t = k[0, t]
                    v_t = v[0, t]
                    if k_t.dim() == 1:
                        n_kv = self.base_model.config.num_key_value_heads
                        d = k_t.shape[0] // n_kv
                        k_t = k_t.view(n_kv, d)
                        v_t = v_t.view(n_kv, d)
                    kv_list_k.append(k_t)
                    kv_list_v.append(v_t)

            if not kv_list_k:
                written_token_indices.pop()
                continue

            kv_dict = {
                "keys": torch.stack(kv_list_k, dim=0),
                "values": torch.stack(kv_list_v, dim=0),
            }

            # Episode routing: accumulate and encode every episode_size tokens
            h_t = first_router_layer_hs[0, t]
            self._episode_buffer.append(h_t)
            self._episode_token_ids.append(input_ids[0, t].item())
            self._episode_token_count += 1

            router_key = None
            episode_text = None
            if self._episode_token_count >= self.config.episode_size:
                ep_hs = torch.stack(self._episode_buffer, dim=0)
                ep_hs = ep_hs.to(next(self.router.parameters()).dtype)
                with torch.no_grad():
                    router_key = self.router.encode_episode(ep_hs)
                if not self._text_used:
                    episode_text = self._current_write_text
                    self._text_used = True
                else:
                    episode_text = None
                self._episode_buffer = []
                self._episode_token_ids = []
                self._episode_token_count = 0

                # Store hidden states for this episode (all layers)
                if hs_buf and router_key is not None:
                    n_all = self.base_model.config.num_hidden_layers
                    ep_start = len(written_token_indices) - self.config.episode_size
                    ep_token_indices = written_token_indices[ep_start:]
                    ep_hs_layers = []
                    for l_idx in range(n_all):
                        if l_idx in hs_buf:
                            hs_l = hs_buf[l_idx][0, ep_token_indices]  # [ep_size, D]
                        else:
                            # Fallback: zeros (shouldn't happen with all-layer interceptor)
                            hs_l = torch.zeros(
                                len(ep_token_indices),
                                self.base_model.config.hidden_size,
                                device=self.device)
                        ep_hs_layers.append(hs_l)
                    ep_hs_tensor = torch.stack(ep_hs_layers, dim=0)  # [n_all, ep_size, D]
                    # Will be stored after store.write sets episode_ptr
                    self._pending_hs = ep_hs_tensor

            self.store.write(kv_dict, router_key, surprise=surprises[0, t].item(),
                            text=episode_text)

            # Write pending hidden states to the episode that was just created
            if hasattr(self, '_pending_hs') and self._pending_hs is not None:
                ep_idx = (self.store.episode_ptr - 1) % (
                    self.config.n_slots // self.config.episode_size)
                self.store.write_hidden_states(ep_idx, self._pending_hs)
                self._pending_hs = None

    def write_memory_full(self, text: str):
        """Write ALL tokens to memory (no surprise filter). For HS injection."""
        if self._is_text_redundant(text):
            return
        self._current_write_text = text
        self._text_used = False
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        T = input_ids.shape[1]
        if T < 2:
            return

        self.interceptor.clear_buffer()

        with torch.no_grad():
            outputs = self.base_model(**inputs, output_hidden_states=True)

        kv_buf = self.interceptor.get_buffered_kv()
        hs_buf = self.interceptor.get_buffered_hidden_states()
        if not kv_buf:
            return

        hidden_states = outputs.hidden_states
        first_hs_idx = min(self.layer_indices[0] + 1, len(hidden_states) - 1)
        first_router_layer_hs = hidden_states[first_hs_idx]

        # Write ALL tokens (no surprise filter)
        for t in range(T):
            kv_list_k = []
            kv_list_v = []
            for li, layer_idx in enumerate(self.layer_indices):
                if layer_idx in kv_buf:
                    k, v = kv_buf[layer_idx]
                    k_t = k[0, t]
                    v_t = v[0, t]
                    if k_t.dim() == 1:
                        n_kv = self.base_model.config.num_key_value_heads
                        d = k_t.shape[0] // n_kv
                        k_t = k_t.view(n_kv, d)
                        v_t = v_t.view(n_kv, d)
                    kv_list_k.append(k_t)
                    kv_list_v.append(v_t)

            if not kv_list_k:
                continue

            kv_dict = {
                "keys": torch.stack(kv_list_k, dim=0),
                "values": torch.stack(kv_list_v, dim=0),
            }

            h_t = first_router_layer_hs[0, t]
            self._episode_buffer.append(h_t)
            self._episode_token_count += 1

            router_key = None
            episode_text = None
            if self._episode_token_count >= self.config.episode_size:
                ep_hs = torch.stack(self._episode_buffer, dim=0)
                ep_hs = ep_hs.to(next(self.router.parameters()).dtype)
                with torch.no_grad():
                    router_key = self.router.encode_episode(ep_hs)
                if not self._text_used:
                    episode_text = self._current_write_text
                    self._text_used = True
                else:
                    episode_text = None
                self._episode_buffer = []
                self._episode_token_count = 0

                # Store hidden states for this episode (all layers, all tokens)
                if hs_buf and router_key is not None:
                    n_all = self.base_model.config.num_hidden_layers
                    ep_start = t - self.config.episode_size + 1
                    ep_token_indices = list(range(max(0, ep_start), t + 1))
                    ep_hs_layers = []
                    for l_idx in range(n_all):
                        if l_idx in hs_buf:
                            ep_hs_layers.append(hs_buf[l_idx][0, ep_token_indices])
                        else:
                            ep_hs_layers.append(torch.zeros(
                                len(ep_token_indices),
                                self.base_model.config.hidden_size,
                                device=self.device))
                    ep_hs_tensor = torch.stack(ep_hs_layers, dim=0)
                    self._pending_hs = ep_hs_tensor

            self.store.write(kv_dict, router_key, surprise=5.0, text=episode_text)

            if hasattr(self, '_pending_hs') and self._pending_hs is not None:
                ep_idx = (self.store.episode_ptr - 1) % (
                    self.config.n_slots // self.config.episode_size)
                self.store.write_hidden_states(ep_idx, self._pending_hs)
                self._pending_hs = None

    def clear_memory(self):
        """Reset memory store."""
        cfg = self.config
        n_kv = self.base_model.config.num_key_value_heads
        self.store = MemoryStore(cfg, len(self.layer_indices), n_kv, self._d_head)
        self._episode_buffer = []
        self._episode_token_ids = []
        self._episode_token_count = 0
        self._current_write_text = None

    # --- Memory mask hooks for non-memory layers ---

    def _install_memory_mask_hooks(self, n_mem_tokens: int) -> list:
        """Install pre-hooks on non-memory decoder layers to mask memory positions.

        Non-memory layers get attention_mask with -inf for the first n_mem_tokens
        positions, so they don't attend to the zero K/V in memory slots.
        Memory layers keep the original mask (attend to real memory K/V).
        """
        hooks = []
        for layer_idx in range(self.base_model.config.num_hidden_layers):
            if layer_idx in self.layer_indices:
                continue  # Memory layer — keep full attention

            layer = self.base_model.model.layers[layer_idx]

            def make_hook(n_mem):
                def hook(module, args, kwargs):
                    mask_key = 'attention_mask'
                    if mask_key in kwargs and kwargs[mask_key] is not None:
                        mask = kwargs[mask_key].clone()
                        # Block memory positions (first n_mem columns)
                        if mask.dtype == torch.bool:
                            mask[:, :, :, :n_mem] = False
                        else:
                            mask[:, :, :, :n_mem] = torch.finfo(mask.dtype).min
                        kwargs[mask_key] = mask
                    return args, kwargs
                return hook

            h = layer.register_forward_pre_hook(make_hook(n_mem_tokens), with_kwargs=True)
            hooks.append(h)
        return hooks

    @staticmethod
    def _remove_hooks(hooks: list):
        for h in hooks:
            h.remove()

    # --- Hidden State Injection (MemoryLLM-style) ---

    def inject_memory_hs(self, query_text: str) -> list | None:
        """Inject memory hidden states via decoder layer hooks.

        For each layer, prepends memory hidden states to the input.
        After attention, strips memory tokens from output.
        Returns list of hooks (caller must remove), or None if no memory.
        """
        router_keys = self.store.get_router_keys()
        if router_keys.shape[0] == 0:
            return None

        # Route query
        q_inputs = self.tokenizer(query_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q_out = self.base_model(**q_inputs, output_hidden_states=True)
        first_hs_idx = min(self.layer_indices[0] + 1, len(q_out.hidden_states) - 1)
        router_dtype = next(self.router.parameters()).dtype
        h_q = q_out.hidden_states[first_hs_idx].to(router_dtype)
        h_pooled = h_q.mean(dim=1, keepdim=True)
        indices, scores = self.router.route(h_pooled, router_keys.to(self.device))
        if indices.shape[-1] == 0:
            return None
        ep_indices = indices[0, 0, :].tolist()

        # Read hidden states for selected episodes
        mem_hs = self.store.read_hidden_states([int(i) for i in ep_indices])
        if mem_hs is None:
            return None
        # mem_hs: [n_all_layers, n_mem_tokens, d_model]
        mem_hs = mem_hs.to(self.device)
        dtype = next(self.base_model.parameters()).dtype
        mem_hs = mem_hs.to(dtype)
        n_mem = mem_hs.shape[1]

        # Install pre-hook (concat) and post-hook (strip) on each decoder layer
        hooks = []
        for layer_idx in range(self.base_model.config.num_hidden_layers):
            layer = self.base_model.model.layers[layer_idx]
            layer_mem = mem_hs[layer_idx].unsqueeze(0)  # [1, n_mem, d_model]

            def make_pre_hook(mem_tokens, rotary_emb):
                def hook(module, args, kwargs):
                    # Get hidden_states from args or kwargs
                    if 'hidden_states' in kwargs:
                        hs = kwargs['hidden_states']
                    else:
                        hs = args[0]
                        args = list(args)

                    # Skip during autoregressive generation (1 token at a time)
                    # Memory influence persists through KV cache from prefill
                    if hs.shape[1] <= 1:
                        return args, kwargs

                    n_m = mem_tokens.shape[1]
                    seq_len = hs.shape[1]
                    total_len = n_m + seq_len

                    # Concatenate memory before input
                    combined = torch.cat([mem_tokens, hs], dim=1)

                    # Extend attention_mask if present
                    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                        mask = kwargs['attention_mask']
                        if mask.dim() == 4:
                            # Build new causal mask for extended sequence
                            fill_val = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
                            allow_val = True if mask.dtype == torch.bool else 0
                            new_mask = torch.full(
                                (mask.shape[0], 1, total_len, total_len),
                                fill_val, device=mask.device, dtype=mask.dtype)
                            # All positions attend to memory (first n_m columns)
                            new_mask[:, :, :, :n_m] = allow_val
                            # Causal mask for query positions (lower triangle)
                            for q in range(n_m, total_len):
                                new_mask[:, :, q, n_m:q+1] = allow_val
                            kwargs['attention_mask'] = new_mask

                    # Recompute position_embeddings for extended sequence
                    new_pos_ids = torch.arange(total_len, device=combined.device).unsqueeze(0)
                    cos, sin = rotary_emb(combined, new_pos_ids)
                    kwargs['position_embeddings'] = (cos, sin)

                    if 'hidden_states' in kwargs:
                        kwargs['hidden_states'] = combined
                    else:
                        args[0] = combined
                        args = tuple(args)

                    return args, kwargs
                return hook

            def make_post_hook(n_m):
                def hook(module, args, kwargs, output):
                    # Strip memory tokens from output
                    if isinstance(output, tuple):
                        hs = output[0]
                        # Only strip if we actually injected (prefill, not generation)
                        if hs.shape[1] > n_m:
                            return (hs[:, n_m:],) + output[1:]
                        return output
                    if output.shape[1] > n_m:
                        return output[:, n_m:]
                    return output
                return hook

            h1 = layer.register_forward_pre_hook(
                make_pre_hook(layer_mem, self.base_model.model.rotary_emb),
                with_kwargs=True)
            h2 = layer.register_forward_hook(
                make_post_hook(n_mem), with_kwargs=True)
            hooks.extend([h1, h2])

        return hooks

    # --- KV Cache Injection with Parallel RoPE ---

    def inject_memory_kv(self, query_text: str) -> DynamicCache | None:
        """Inject memory K/V into DynamicCache with Parallel RoPE.

        Returns a DynamicCache that can be passed as past_key_values.
        The caller must extend attention_mask and offset position_ids
        by cache.get_seq_length().
        """
        router_keys = self.store.get_router_keys()
        if router_keys.shape[0] == 0:
            return None

        # Route query to find relevant episodes
        q_inputs = self.tokenizer(query_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q_out = self.base_model(**q_inputs, output_hidden_states=True)
        first_hs_idx = min(self.layer_indices[0] + 1, len(q_out.hidden_states) - 1)
        router_dtype = next(self.router.parameters()).dtype
        h_q = q_out.hidden_states[first_hs_idx].to(router_dtype)
        h_pooled = h_q.mean(dim=1, keepdim=True)
        indices, scores = self.router.route(h_pooled, router_keys.to(self.device))
        if indices.shape[-1] == 0:
            return None
        ep_indices = indices[0, 0, :].tolist()

        # Read pre-RoPE K/V from store
        result = self.store.read([int(i) for i in ep_indices])
        if result is None:
            return None
        mem_k, mem_v = result  # [n_router_layers, n_tokens, n_kv_heads, d_head]

        # Parallel RoPE: memory tokens get independent positions [0, 1, ..., N-1]
        n_mem_tokens = mem_k.shape[1]
        mem_pos_ids = torch.arange(n_mem_tokens, device=self.device).unsqueeze(0)

        # Get RoPE cos/sin from the model's rotary embedding
        rotary_emb = self.base_model.model.rotary_emb
        cos, sin = rotary_emb(
            mem_k.new_zeros(1, n_mem_tokens, self.base_model.config.hidden_size).to(self.device),
            mem_pos_ids,
        )

        # Import apply_rotary_pos_emb from the model's module
        model_module = type(self.base_model).__module__
        import importlib
        modeling_mod = importlib.import_module(model_module)
        apply_rotary = modeling_mod.apply_rotary_pos_emb

        cache = DynamicCache()
        n_total_layers = self.base_model.config.num_hidden_layers
        n_kv_heads = self.base_model.config.num_key_value_heads
        d_head = self._d_head
        dtype = next(self.base_model.parameters()).dtype

        for layer_idx in range(n_total_layers):
            if layer_idx in self.layer_indices:
                li = self.layer_indices.index(layer_idx)
                k = mem_k[li].unsqueeze(0).to(self.device)  # [1, n_tokens, n_kv_heads, d_head]
                v = mem_v[li].unsqueeze(0).to(self.device)

                # Transpose to [1, n_kv_heads, n_tokens, d_head] (attention format)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                # Apply k_norm if present (Qwen3 has QKNorm, Qwen2.5 doesn't)
                attn = self.base_model.model.layers[layer_idx].self_attn
                if hasattr(attn, 'k_norm'):
                    k = attn.k_norm(k)

                # Apply RoPE to K only (V never gets RoPE)
                k, _ = apply_rotary(k, k, cos, sin)

                cache.update(k.to(dtype), v.to(dtype), layer_idx)
            else:
                # Zero K/V for non-memory layers (same length for consistent attention_mask)
                cache.update(
                    torch.zeros(1, n_kv_heads, n_mem_tokens, d_head, device=self.device, dtype=dtype),
                    torch.zeros(1, n_kv_heads, n_mem_tokens, d_head, device=self.device, dtype=dtype),
                    layer_idx,
                )

        return cache

    def generate_with_memory(self, query_text: str, max_new_tokens: int = 30,
                              do_sample: bool = False, **gen_kwargs) -> str:
        """Generate with hybrid MSA: text prefix + KV cache injection.

        Memory layers attend to real memory K/V. Non-memory layers have
        memory positions masked out (-inf) so zero K/V don't affect them.
        """
        # Text prefix (retrieve memory texts into prompt)
        prompt = self.retrieve_and_format(query_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # KV cache injection
        mem_cache = self.inject_memory_kv(query_text)

        gen_args = dict(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        hooks = []
        if mem_cache is not None:
            n_mem = mem_cache.get_seq_length()
            seq_len = inputs["input_ids"].shape[1]
            gen_args["past_key_values"] = mem_cache
            gen_args["attention_mask"] = torch.ones(
                1, n_mem + seq_len, device=self.device, dtype=torch.long)
            gen_args["position_ids"] = torch.arange(
                n_mem, n_mem + seq_len, device=self.device).unsqueeze(0)
            gen_args["cache_position"] = torch.arange(
                n_mem, n_mem + seq_len, device=self.device)
            # Mask memory positions in non-memory layers
            hooks = self._install_memory_mask_hooks(n_mem)

        try:
            with torch.no_grad():
                gen_ids = self.base_model.generate(**gen_args)
        finally:
            self._remove_hooks(hooks)

        answer_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    # --- Text prefix injection (MSA-inspired) ---

    def retrieve_and_format(self, query_text: str) -> str:
        """Route query through router, retrieve memory texts, format prompt."""
        router_keys = self.store.get_router_keys()
        if router_keys.shape[0] == 0:
            return self._format_query_only(query_text)

        # Get hidden states for routing
        q_inputs = self.tokenizer(query_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q_out = self.base_model(**q_inputs, output_hidden_states=True)
        first_hs_idx = min(self.layer_indices[0] + 1, len(q_out.hidden_states) - 1)
        router_dtype = next(self.router.parameters()).dtype
        h_q = q_out.hidden_states[first_hs_idx].to(router_dtype)  # [1, T, D]
        h_pooled = h_q.mean(dim=1, keepdim=True)  # [1, 1, D]

        indices, scores = self.router.route(h_pooled, router_keys.to(self.device))
        if indices.shape[-1] == 0:
            return self._format_query_only(query_text)

        ep_indices = indices[0, 0, :].tolist()
        texts = self.store.read_texts([int(i) for i in ep_indices])
        if not texts:
            return self._format_query_only(query_text)

        return self._format_with_memories(query_text, texts)

    def _format_with_memories(self, query_text: str, memory_texts: list[str]) -> str:
        """Format prompt with memories in system prompt (instruct chat template)."""
        system_parts = ["You have access to memories. Use them to answer accurately."]
        for i, text in enumerate(memory_texts, 1):
            system_parts.append(f"Memory {i}: {text}")
        system_msg = "\n".join(system_parts)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _format_query_only(self, query_text: str) -> str:
        """Format prompt without memories."""
        messages = [{"role": "user", "content": query_text}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_contrastive_loss(wrapper: CellMemWrapper, batch: list[dict]) -> torch.Tensor:
    """CLIP-style InfoNCE on a batch of (memory, query) pairs."""
    B = len(batch)
    q_vecs = []
    k_vecs = []

    for item in batch:
        # Encode memory episode → router key
        mem_inputs = wrapper.tokenizer(item["memory"], return_tensors="pt").to(wrapper.device)
        with torch.no_grad():
            mem_out = wrapper.base_model(**mem_inputs, output_hidden_states=True)
        first_hs_idx = min(wrapper.layer_indices[0] + 1, len(mem_out.hidden_states) - 1)
        router_dtype = next(wrapper.router.parameters()).dtype
        hs = mem_out.hidden_states[first_hs_idx][0].to(router_dtype)  # [T, D]
        k = wrapper.router.encode_episode(hs)  # [d_router]

        # Encode query → router query vector
        q_inputs = wrapper.tokenizer(item["query"], return_tensors="pt").to(wrapper.device)
        with torch.no_grad():
            q_out = wrapper.base_model(**q_inputs, output_hidden_states=True)
        first_hs_idx_q = min(wrapper.layer_indices[0] + 1, len(q_out.hidden_states) - 1)
        h_q = q_out.hidden_states[first_hs_idx_q][0].mean(0).to(router_dtype)  # [D]
        q = wrapper.router.w_q_r(h_q)  # [d_router]

        q_vecs.append(q)
        k_vecs.append(k)

    q_batch = torch.stack(q_vecs)   # [B, d_router]
    k_batch = torch.stack(k_vecs)   # [B, d_router]

    # For each query, its own episode is positive; all others are negatives (CLIP / InfoNCE)
    losses = []
    for i in range(B):
        pos = k_batch[i]                          # [d_router]
        neg_mask = torch.ones(B, dtype=torch.bool)
        neg_mask[i] = False
        negs = k_batch[neg_mask].unsqueeze(0)     # [1, B-1, d_router]
        q_i = q_batch[i].unsqueeze(0)             # [1, d_router]
        pos_i = pos.unsqueeze(0)                  # [1, d_router]
        loss = wrapper.router.contrastive_loss(q_i, pos_i, negs, tau=wrapper.config.contrastive_tau)
        losses.append(loss)
    return torch.stack(losses).mean()


@contextlib.contextmanager
def maybe_autocast(device: str):
    """Context manager for bfloat16 autocast on CUDA, no-op on CPU."""
    if device != "cpu" and torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def compute_lm_loss(wrapper: CellMemWrapper, batch: list[dict]) -> torch.Tensor:
    """Language model loss: write memory, retrieve text, predict answer."""
    losses = []
    for item in batch:
        wrapper.clear_memory()
        wrapper.write_memory(item["memory"])

        # Route + retrieve text → format prompt with memory prefix
        query_with_answer = f"Question: {item['query']}\nAnswer: {item['answer']}"
        prompt = wrapper.retrieve_and_format(query_with_answer)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)
        targets = inputs["input_ids"].clone()
        with maybe_autocast(wrapper.device):
            outputs = wrapper.base_model(**inputs)
        logits = outputs.logits  # [1, T, V]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            targets[:, 1:].reshape(-1),
        )
        losses.append(loss)
        wrapper.clear_memory()

    return torch.stack(losses).mean()


def compute_kv_lm_loss(wrapper: CellMemWrapper, batch: list[dict]) -> torch.Tensor:
    """LM loss with KV injection: model must use injected K/V to predict answer.

    No text prefix — the model can only access memory through the injected K/V.
    Gradient flows through LoRA on q_proj of memory layers.
    """
    losses = []
    for item in batch:
        wrapper.clear_memory()
        wrapper.write_memory(item["memory"])

        # KV injection (no text prefix)
        q_text = f"Question: {item['query']}\nAnswer: {item['answer']}"
        mem_cache = wrapper.inject_memory_kv(q_text)
        prompt = wrapper._format_query_only(q_text)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)
        targets = inputs["input_ids"].clone()

        if mem_cache is not None:
            n_mem = mem_cache.get_seq_length()
            seq_len = inputs["input_ids"].shape[1]
            attn_mask = torch.ones(1, n_mem + seq_len, device=wrapper.device, dtype=torch.long)
            pos_ids = torch.arange(n_mem, n_mem + seq_len, device=wrapper.device).unsqueeze(0)
            hooks = wrapper._install_memory_mask_hooks(n_mem)
            try:
                with maybe_autocast(wrapper.device):
                    outputs = wrapper.base_model(
                        input_ids=inputs["input_ids"],
                        past_key_values=mem_cache,
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                    )
            finally:
                wrapper._remove_hooks(hooks)
        else:
            with maybe_autocast(wrapper.device):
                outputs = wrapper.base_model(**inputs)

        logits = outputs.logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            targets[:, 1:].reshape(-1),
        )
        losses.append(loss)
        wrapper.clear_memory()

    return torch.stack(losses).mean()


def compute_hs_lm_loss(wrapper: CellMemWrapper, batch: list[dict]) -> torch.Tensor:
    """LM loss with hidden state injection (MemoryLLM-style).

    Model must use injected hidden states to predict answer.
    Gradient flows through LoRA on all layers.
    """
    losses = []
    for item in batch:
        wrapper.clear_memory()
        wrapper.write_memory_full(item["memory"])

        q_text = f"Question: {item['query']}\nAnswer: {item['answer']}"
        hooks = wrapper.inject_memory_hs(q_text)
        prompt = wrapper._format_query_only(q_text)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)
        targets = inputs["input_ids"].clone()

        try:
            with maybe_autocast(wrapper.device):
                outputs = wrapper.base_model(input_ids=inputs["input_ids"])
        finally:
            if hooks:
                wrapper._remove_hooks(hooks)

        logits = outputs.logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            targets[:, 1:].reshape(-1),
        )
        losses.append(loss)
        wrapper.clear_memory()

    return torch.stack(losses).mean()


def eval_router(wrapper: CellMemWrapper, data: list[dict]) -> dict:
    """Evaluate router recall and discrimination gap.

    Only items where routing was possible (active_episodes > 0) are counted
    in the recall denominator. Items with no episodes are silently skipped.
    """
    recall_hits = 0
    routable_count = 0
    skipped_count = 0
    pos_scores = []
    neg_scores = []

    for item in data[:50]:
        wrapper.clear_memory()
        wrapper.write_memory(item["memory"])

        router_keys = wrapper.store.get_router_keys()
        if router_keys.shape[0] == 0:
            skipped_count += 1
            continue

        routable_count += 1
        q_inputs = wrapper.tokenizer(item["query"], return_tensors="pt").to(wrapper.device)
        with torch.no_grad():
            q_out = wrapper.base_model(**q_inputs, output_hidden_states=True)
        first_hs_idx = min(wrapper.layer_indices[0] + 1, len(q_out.hidden_states) - 1)
        router_dtype = next(wrapper.router.parameters()).dtype
        h_q = q_out.hidden_states[first_hs_idx][0].mean(0).to(router_dtype)
        q = F.normalize(wrapper.router.w_q_r(h_q), dim=-1)

        router_keys_norm = F.normalize(router_keys.to(wrapper.device), dim=-1)
        sims = (q @ router_keys_norm.T)
        n_eps = sims.shape[0]
        k = min(wrapper.config.top_k, n_eps)
        top_indices = sims.topk(k).indices.tolist()

        # Episode 0 is "correct" in our simple test (first memory written)
        if 0 in top_indices:
            recall_hits += 1

        pos_scores.append(sims[0].item())
        if n_eps > 1:
            neg_scores.extend(sims[1:].tolist())

    print(f"  [eval_router] routable={routable_count}, skipped(no episode)={skipped_count}")
    n = max(routable_count, 1)
    gap = (sum(pos_scores) / max(len(pos_scores), 1) -
           sum(neg_scores) / max(len(neg_scores), 1))
    return {
        "recall_at_k": recall_hits / n,
        "discrimination_gap": gap,
        "routable_fraction": routable_count / max(routable_count + skipped_count, 1),
    }


def eval_generation(wrapper: CellMemWrapper, data: list[dict],
                    mode: str = "text_only", debug: bool = False) -> dict:
    """Evaluate generation recall with different memory injection modes.

    Modes:
        text_only: Text prefix injection only (baseline)
        kv_only:   KV cache injection only (no text in prompt)
        hybrid:    Text prefix + KV cache injection (MSA approach)
        hs_only:   Hidden state injection only (MemoryLLM-style)
    """
    correct = 0
    n = min(20, len(data))

    for item in data[:n]:
        wrapper.clear_memory()
        if mode == "hs_only":
            wrapper.write_memory_full(item["memory"])
        else:
            wrapper.write_memory(item["memory"])
        has_memory = wrapper.store.active_episodes > 0

        q_text = f"Question: {item['query']}\nAnswer:"

        if mode == "hybrid":
            generated = wrapper.generate_with_memory(q_text, max_new_tokens=30)
        elif mode == "hs_only":
            # Hidden state injection: prefill with hooks → cache → generate without hooks
            hooks = wrapper.inject_memory_hs(q_text)
            prompt = wrapper._format_query_only(q_text)
            inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)
            if hooks:
                # Prefill: run forward with hooks to populate KV cache
                with torch.no_grad():
                    prefill_out = wrapper.base_model(
                        input_ids=inputs["input_ids"],
                        use_cache=True,
                    )
                wrapper._remove_hooks(hooks)
                # Generate from cache (no hooks needed)
                past_kv = prefill_out.past_key_values
                last_token = prefill_out.logits[:, -1:].argmax(dim=-1)
                generated_ids = [last_token]
                for _ in range(29):
                    with torch.no_grad():
                        out = wrapper.base_model(
                            input_ids=last_token,
                            past_key_values=past_kv,
                            use_cache=True,
                        )
                    past_kv = out.past_key_values
                    last_token = out.logits[:, -1:].argmax(dim=-1)
                    generated_ids.append(last_token)
                    if last_token.item() == wrapper.tokenizer.eos_token_id:
                        break
                gen_tokens = torch.cat(generated_ids, dim=1)
                generated = wrapper.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
            else:
                with torch.no_grad():
                    gen_ids = wrapper.base_model.generate(
                        input_ids=inputs["input_ids"],
                        max_new_tokens=30, do_sample=False,
                        pad_token_id=wrapper.tokenizer.eos_token_id,
                    )
                answer_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
                generated = wrapper.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        elif mode == "kv_only":
            # KV cache injection with plain query (no text prefix)
            mem_cache = wrapper.inject_memory_kv(q_text)
            prompt = wrapper._format_query_only(q_text)
            inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)

            gen_args = dict(
                input_ids=inputs["input_ids"],
                max_new_tokens=30, do_sample=False,
                pad_token_id=wrapper.tokenizer.eos_token_id,
            )
            hooks = []
            if mem_cache is not None:
                n_mem = mem_cache.get_seq_length()
                seq_len = inputs["input_ids"].shape[1]
                gen_args["past_key_values"] = mem_cache
                gen_args["attention_mask"] = torch.ones(
                    1, n_mem + seq_len, device=wrapper.device, dtype=torch.long)
                gen_args["position_ids"] = torch.arange(
                    n_mem, n_mem + seq_len, device=wrapper.device).unsqueeze(0)
                gen_args["cache_position"] = torch.arange(
                    n_mem, n_mem + seq_len, device=wrapper.device)
                hooks = wrapper._install_memory_mask_hooks(n_mem)

            try:
                with torch.no_grad():
                    gen_ids = wrapper.base_model.generate(**gen_args)
            finally:
                wrapper._remove_hooks(hooks)
            answer_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
            generated = wrapper.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        else:  # text_only
            prompt = wrapper.retrieve_and_format(q_text)
            inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)
            with torch.no_grad():
                gen_ids = wrapper.base_model.generate(
                    **inputs, max_new_tokens=30, do_sample=False,
                    pad_token_id=wrapper.tokenizer.eos_token_id,
                )
            answer_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
            generated = wrapper.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        hit = item["answer"].lower() in generated.lower()
        if hit:
            correct += 1

        if debug:
            status = "✓" if hit else "✗"
            print(f"  {status} [{mode}] mem={has_memory} | q: {item['query'][:40]!r}")
            print(f"      expected={item['answer']!r} | got={generated[:50]!r}")

        wrapper.clear_memory()

    return {"generation_recall": correct / max(n, 1)}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_phase(
    wrapper: CellMemWrapper,
    data: list[dict],
    n_epochs: int,
    lm_weight: float,
    contrastive_weight: float,
    lr: float = 1e-3,
    batch_size: int = 8,
    phase_name: str = "phase",
):
    optimizer = torch.optim.AdamW(wrapper.trainable_params(), lr=lr)
    random.shuffle(data)

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if len(batch) < 2:
                continue

            optimizer.zero_grad()
            loss_c = compute_contrastive_loss(wrapper, batch)
            loss_lm = compute_lm_loss(wrapper, batch)
            loss = contrastive_weight * loss_c + lm_weight * loss_lm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.trainable_params(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"[{phase_name}] epoch {epoch+1}/{n_epochs} | loss={avg:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--epochs1", type=int, default=20)
    parser.add_argument("--epochs2", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory to save router checkpoints after each phase")
    parser.add_argument("--resume-phase2", default=None, metavar="CKPT",
                        help="Skip Phase 1 and load router checkpoint, then run Phase 2")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank for memory layer q_proj")
    parser.add_argument("--lora-epochs", type=int, default=10,
                        help="Number of epochs for LoRA training (Phase 3)")
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to(args.device)

    n_layers = model.config.num_hidden_layers
    layer_indices = list(range(n_layers - 4, n_layers))
    print(f"Router layers: {layer_indices}")

    config = CellMemConfig(
        router_layers=layer_indices,
        episode_size=8,
        top_k=4,
        surprise_threshold=2.0,  # lowered from 4.0: more tokens stored → more episodes
    )
    wrapper = CellMemWrapper(model, tokenizer, layer_indices, config=config, device=args.device)

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_data = generate_training_data(args.n_train)
    eval_data = generate_training_data(20)

    if args.resume_phase2:
        print(f"\n=== Resuming: loading router from {args.resume_phase2} ===")
        state = torch.load(args.resume_phase2, map_location=args.device, weights_only=True)
        wrapper.router.load_state_dict(state)
    else:
        # Phase 1: warmup router
        print("\n=== Phase 1: Router warmup ===")
        train_phase(wrapper, train_data, n_epochs=args.epochs1,
                    lm_weight=0.1, contrastive_weight=1.0,
                    lr=args.lr, batch_size=args.batch_size, phase_name="p1")

        metrics = eval_router(wrapper, eval_data)
        print(f"\nPhase 1 eval: router_recall@{config.top_k}={metrics['recall_at_k']:.2%}, "
              f"discrimination_gap={metrics['discrimination_gap']:.3f}")
        if metrics["discrimination_gap"] < 0.1:
            print("WARNING: Router not discriminating (gap < 0.1). Check training.")

        if ckpt_dir:
            p1_path = ckpt_dir / "router_phase1.pt"
            torch.save(wrapper.router.state_dict(), p1_path)
            # Also save as latest for preemption recovery
            torch.save(wrapper.router.state_dict(), ckpt_dir / "router_latest.pt")
            print(f"Router checkpoint saved: {p1_path}")

    # Phase 2: generation validation — 3-way comparison
    print("\n=== Phase 2: Generation Validation ===")
    metrics_final = eval_router(wrapper, eval_data)

    print("\n--- text_only (baseline) ---")
    gen_text = eval_generation(wrapper, eval_data, mode="text_only", debug=True)
    print(f"  text_only recall: {gen_text['generation_recall']:.2%}")

    print("\n--- kv_only (KV cache injection) ---")
    gen_kv = eval_generation(wrapper, eval_data, mode="kv_only", debug=True)
    print(f"  kv_only recall: {gen_kv['generation_recall']:.2%}")

    print("\n--- hybrid (text + KV injection) ---")
    gen_hybrid = eval_generation(wrapper, eval_data, mode="hybrid", debug=True)
    print(f"  hybrid recall: {gen_hybrid['generation_recall']:.2%}")

    print("\n--- hs_only (hidden state injection) ---")
    gen_hs = eval_generation(wrapper, eval_data, mode="hs_only", debug=True)
    print(f"  hs_only recall: {gen_hs['generation_recall']:.2%}")

    # Additional training if text_only baseline too low
    if gen_text["generation_recall"] < 0.75 and args.epochs2 > 0:
        print(f"\nText-only recall {gen_text['generation_recall']:.2%} < 75%, "
              "running additional router training...")
        train_phase(wrapper, train_data, n_epochs=args.epochs2,
                    lm_weight=0.0, contrastive_weight=1.0,
                    lr=args.lr * 0.3, batch_size=args.batch_size, phase_name="p2")
        metrics_final = eval_router(wrapper, eval_data)
        gen_text = eval_generation(wrapper, eval_data, mode="text_only", debug=True)
        gen_kv = eval_generation(wrapper, eval_data, mode="kv_only", debug=True)
        gen_hybrid = eval_generation(wrapper, eval_data, mode="hybrid", debug=True)

    print(f"\nFinal eval:")
    print(f"  router_recall@{config.top_k}={metrics_final['recall_at_k']:.2%} (target >=80%)")
    print(f"  discrimination_gap={metrics_final['discrimination_gap']:.3f} (target >=0.3)")
    print(f"  text_only_recall={gen_text['generation_recall']:.2%} (baseline)")
    print(f"  kv_only_recall={gen_kv['generation_recall']:.2%} (target >=50%)")
    print(f"  hybrid_recall={gen_hybrid['generation_recall']:.2%} (target >=85%)")

    # Phase 3: LoRA training on ALL layers (teach attention to read memory hidden states)
    if args.lora_epochs > 0:
        print(f"\n=== Phase 3: LoRA + HS injection (rank={args.lora_rank}, {args.lora_epochs} epochs) ===")
        wrapper.install_memory_lora(rank=args.lora_rank, all_layers=True)
        lora_params = sum(p.numel() for p in wrapper.trainable_params()
                         if p.requires_grad)
        print(f"  Trainable params: {lora_params:,} (router + LoRA)")

        # Use conversational + factual training data
        conv_data = generate_conversational_data(200)
        print(f"  Training samples: {len(conv_data)} (conversational + factual)")

        optimizer = torch.optim.AdamW(wrapper.trainable_params(), lr=args.lr * 0.1)
        for epoch in range(args.lora_epochs):
            random.shuffle(conv_data)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(conv_data), args.batch_size):
                batch = conv_data[i:i + args.batch_size]
                if len(batch) < 2:
                    continue
                optimizer.zero_grad()
                loss = compute_hs_lm_loss(wrapper, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wrapper.trainable_params(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg = total_loss / max(n_batches, 1)
            print(f"[p3-lora] epoch {epoch+1}/{args.lora_epochs} | hs_lm_loss={avg:.4f}")

        # Eval after LoRA training
        print("\n--- hs_only after LoRA ---")
        gen_hs_lora = eval_generation(wrapper, eval_data, mode="hs_only", debug=True)
        print(f"  hs_only recall (with LoRA): {gen_hs_lora['generation_recall']:.2%}")

        if ckpt_dir:
            # Save LoRA state
            lora_state = {}
            for layer_idx in wrapper.layer_indices:
                attn = wrapper.base_model.model.layers[layer_idx].self_attn
                if hasattr(attn, '_original_q_proj'):
                    lora = attn.q_proj
                    lora_state[f"layer_{layer_idx}_lora_A"] = lora.lora_A.state_dict()
                    lora_state[f"layer_{layer_idx}_lora_B"] = lora.lora_B.state_dict()
            lora_path = ckpt_dir / "lora_final.pt"
            torch.save(lora_state, lora_path)
            print(f"LoRA checkpoint saved: {lora_path}")

    if ckpt_dir:
        final_path = ckpt_dir / "router_final.pt"
        torch.save(wrapper.router.state_dict(), final_path)
        torch.save(wrapper.router.state_dict(), ckpt_dir / "router_latest.pt")
        print(f"Final router checkpoint saved: {final_path}")


if __name__ == "__main__":
    main()
