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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanochat.cellmem_v3 import CellMemConfig, MemoryRouter, MemoryStore
from nanochat.cellmem_v2 import SurpriseCalculator
from nanochat.kv_interceptor import KVInterceptor


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


def generate_training_data(n: int = 100) -> list[dict]:
    data = []
    facts = FICTIONAL_FACTS * (n // len(FICTIONAL_FACTS) + 1)
    random.shuffle(facts)
    for memory, query, answer in facts[:n]:
        data.append({"memory": memory, "query": query, "answer": answer})
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
        self.interceptor = KVInterceptor(model, layer_indices, model_type="qwen")
        self._episode_buffer: list = []  # hidden states for current episode
        self._episode_token_count = 0
        self._episode_token_ids: list = []  # token ids for current episode (for text decoding)
        self._current_write_text: str | None = None

        # Freeze backbone
        for p in model.parameters():
            p.requires_grad_(False)

    def trainable_params(self):
        return list(self.router.parameters())

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
        if not kv_buf:
            return

        # hidden_states is a tuple of (n_hidden_layers+1) tensors; index 0 = embedding output,
        # index i = output of layer i-1. Use layer index + 1 to get post-layer hidden states.
        hidden_states = outputs.hidden_states  # tuple of [1, T, D] per layer (+1 for embedding)
        first_hs_idx = min(self.layer_indices[0] + 1, len(hidden_states) - 1)
        first_router_layer_hs = hidden_states[first_hs_idx]  # [1, T, D]

        for t in range(T):
            if not write_mask[0, t].item():
                continue

            # Build per-layer K/V dict for this token
            kv_list_k = []
            kv_list_v = []
            for li, layer_idx in enumerate(self.layer_indices):
                if layer_idx in kv_buf:
                    k, v = kv_buf[layer_idx]  # [1, T, n_kv_heads * d_head]
                    # Slice token t
                    k_t = k[0, t]  # [n_kv_heads * d_head]
                    v_t = v[0, t]
                    # Reshape to [n_kv_heads, d_head]
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
                "keys": torch.stack(kv_list_k, dim=0),    # [n_layers, n_kv_heads, d_head]
                "values": torch.stack(kv_list_v, dim=0),
            }

            # Episode routing: accumulate and encode every episode_size tokens
            h_t = first_router_layer_hs[0, t]  # [d_model]
            self._episode_buffer.append(h_t)
            self._episode_token_ids.append(input_ids[0, t].item())
            self._episode_token_count += 1

            router_key = None
            episode_text = None
            if self._episode_token_count >= self.config.episode_size:
                ep_hs = torch.stack(self._episode_buffer, dim=0)  # [episode_size, d_model]
                # Cast to router dtype (float32) in case backbone uses bfloat16
                ep_hs = ep_hs.to(next(self.router.parameters()).dtype)
                with torch.no_grad():
                    router_key = self.router.encode_episode(ep_hs)
                # Attach full text only to the first episode of this write_memory() call
                if not self._text_used:
                    episode_text = self._current_write_text
                    self._text_used = True
                else:
                    episode_text = None
                self._episode_buffer = []
                self._episode_token_ids = []
                self._episode_token_count = 0

            self.store.write(kv_dict, router_key, surprise=surprises[0, t].item(),
                            text=episode_text)

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
    """
    correct = 0
    n = min(20, len(data))

    for item in data[:n]:
        wrapper.clear_memory()
        wrapper.write_memory(item["memory"])
        has_memory = wrapper.store.active_episodes > 0

        q_text = f"Question: {item['query']}\nAnswer:"

        if mode == "hybrid":
            generated = wrapper.generate_with_memory(q_text, max_new_tokens=30)
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

    if ckpt_dir:
        final_path = ckpt_dir / "router_final.pt"
        torch.save(wrapper.router.state_dict(), final_path)
        torch.save(wrapper.router.state_dict(), ckpt_dir / "router_latest.pt")
        print(f"Final router checkpoint saved: {final_path}")


if __name__ == "__main__":
    main()
