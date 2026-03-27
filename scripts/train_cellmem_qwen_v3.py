# scripts/train_cellmem_qwen_v3.py
"""
CellMem v3 Training Script.

Two-phase training: router warmup (InfoNCE) then generation validation.
Only router (W_Q^R, W_K^R) is trained. Backbone always frozen.

Usage:
    python scripts/train_cellmem_qwen_v3.py --model Qwen/Qwen2.5-0.5B --epochs1 20 --epochs2 30
"""
from __future__ import annotations
import argparse
import contextlib
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        d_head = d_model // model.config.num_attention_heads

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
        self._read_hooks: list = []
        self._episode_buffer: list = []  # hidden states for current episode
        self._episode_token_count = 0

        # Freeze backbone
        for p in model.parameters():
            p.requires_grad_(False)

    def trainable_params(self):
        return list(self.router.parameters())

    def _install_read_hooks(self):
        """Install hooks that inject memory into the residual stream during forward."""
        self._remove_read_hooks()
        for layer_pos, layer_idx in enumerate(self.layer_indices):
            attn = self.base_model.model.layers[layer_idx].self_attn

            def make_hook(lpos):
                def hook(module, args, kwargs):
                    # args[0] is hidden_states in most HF Qwen implementations
                    if isinstance(args, tuple) and len(args) > 0:
                        h = args[0]
                    else:
                        return args, kwargs

                    router_keys = self.store.get_router_keys()
                    if router_keys.shape[0] == 0:
                        return args, kwargs

                    router_keys = router_keys.to(self.device)
                    indices, scores = self.router.route(h, router_keys)
                    if indices.shape[-1] == 0:
                        return args, kwargs

                    # Use mean over batch/time for episode selection: take first batch, first token
                    ep_indices = indices[0, 0, :].tolist()  # [top_k] episode indices
                    result = self.store.read([int(i) for i in ep_indices])
                    if result is None:
                        return args, kwargs

                    k_mem, v_mem = result  # [n_layers, n_tokens, n_kv_heads, d_head]
                    k_layer = k_mem[lpos].to(self.device)  # [n_tokens, n_kv_heads, d_head]
                    v_layer = v_mem[lpos].to(self.device)

                    # Attend: Q from backbone q_proj (frozen), K/V from memory (no RoPE)
                    B, T, D = h.shape
                    q = module.q_proj(h)  # [B, T, n_heads * d_head]

                    n_heads = self.base_model.config.num_attention_heads
                    n_kv_heads_cfg = self.base_model.config.num_key_value_heads
                    d_head = D // n_heads

                    # Reshape Q for multi-head attention
                    q = q.view(B, T, n_heads, d_head).transpose(1, 2)  # [B, n_heads, T, d_head]

                    N_mem = k_layer.shape[0]
                    # k_layer: [N_mem, n_kv_heads, d_head] → [B, n_kv_heads, N_mem, d_head]
                    k_m = k_layer.unsqueeze(0).expand(B, -1, -1, -1).permute(0, 2, 1, 3)
                    v_m = v_layer.unsqueeze(0).expand(B, -1, -1, -1).permute(0, 2, 1, 3)

                    # Expand KV heads to match Q heads (GQA)
                    if n_kv_heads_cfg < n_heads:
                        repeat = n_heads // n_kv_heads_cfg
                        k_m = k_m.repeat_interleave(repeat, dim=1)
                        v_m = v_m.repeat_interleave(repeat, dim=1)

                    # Cast memory tensors to match h dtype
                    k_m = k_m.to(h.dtype)
                    v_m = v_m.to(h.dtype)

                    # Scaled dot-product attention (no RoPE on memory K/V)
                    attn_out = F.scaled_dot_product_attention(q, k_m, v_m)  # [B, n_heads, T, d_head]
                    attn_out = attn_out.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
                    attn_out = module.o_proj(attn_out)  # [B, T, D]

                    # Residual update
                    new_h = h + attn_out
                    return (new_h,) + args[1:], kwargs

                return hook

            h = attn.register_forward_pre_hook(make_hook(layer_pos), with_kwargs=True)
            self._read_hooks.append(h)

    def _remove_read_hooks(self):
        for h in self._read_hooks:
            h.remove()
        self._read_hooks.clear()

    def write_memory(self, text: str):
        """Process text and write surprising tokens to memory store."""
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
            self._episode_token_count += 1

            router_key = None
            if self._episode_token_count >= self.config.episode_size:
                ep_hs = torch.stack(self._episode_buffer, dim=0)  # [episode_size, d_model]
                # Cast to router dtype (float32) in case backbone uses bfloat16
                ep_hs = ep_hs.to(next(self.router.parameters()).dtype)
                with torch.no_grad():
                    router_key = self.router.encode_episode(ep_hs)
                self._episode_buffer = []
                self._episode_token_count = 0

            self.store.write(kv_dict, router_key, surprise=surprises[0, t].item())

    def clear_memory(self):
        """Reset memory store."""
        cfg = self.config
        n_kv = self.base_model.config.num_key_value_heads
        d_head = (self.base_model.config.hidden_size //
                  self.base_model.config.num_attention_heads)
        self.store = MemoryStore(cfg, len(self.layer_indices), n_kv, d_head)
        self._episode_buffer = []
        self._episode_token_count = 0


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
    """Language model loss: write memory, then predict answer given query."""
    losses = []
    wrapper._install_read_hooks()
    for item in batch:
        wrapper.clear_memory()
        wrapper.write_memory(item["memory"])

        # Query + answer as target
        full_text = item["query"] + " " + item["answer"]
        inputs = wrapper.tokenizer(full_text, return_tensors="pt").to(wrapper.device)
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

    wrapper._remove_read_hooks()
    return torch.stack(losses).mean()


def eval_router(wrapper: CellMemWrapper, data: list[dict]) -> dict:
    """Evaluate router recall and discrimination gap."""
    recall_hits = 0
    pos_scores = []
    neg_scores = []

    for item in data[:50]:
        wrapper.clear_memory()
        wrapper.write_memory(item["memory"])

        router_keys = wrapper.store.get_router_keys()
        if router_keys.shape[0] == 0:
            continue

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

    n = max(len(data[:50]), 1)
    gap = (sum(pos_scores) / max(len(pos_scores), 1) -
           sum(neg_scores) / max(len(neg_scores), 1))
    return {
        "recall_at_k": recall_hits / n,
        "discrimination_gap": gap,
    }


def eval_generation(wrapper: CellMemWrapper, data: list[dict]) -> dict:
    """Evaluate generation recall: does model generate the correct answer?"""
    wrapper._install_read_hooks()
    correct = 0
    n = min(20, len(data))  # quick eval

    for item in data[:n]:
        wrapper.clear_memory()
        wrapper.write_memory(item["memory"])

        q_inputs = wrapper.tokenizer(item["query"], return_tensors="pt").to(wrapper.device)
        with torch.no_grad():
            gen_ids = wrapper.base_model.generate(
                **q_inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=wrapper.tokenizer.eos_token_id,
            )
        answer_ids = gen_ids[0, q_inputs["input_ids"].shape[1]:]
        generated = wrapper.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        if item["answer"].lower() in generated.lower():
            correct += 1
        wrapper.clear_memory()

    wrapper._remove_read_hooks()
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs1", type=int, default=20)
    parser.add_argument("--epochs2", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to(args.device)

    n_layers = model.config.num_hidden_layers
    # Default: top-4 layers
    layer_indices = list(range(n_layers - 4, n_layers))
    print(f"Router layers: {layer_indices}")

    config = CellMemConfig(
        router_layers=layer_indices,
        episode_size=8,
        top_k=4,
    )
    wrapper = CellMemWrapper(model, tokenizer, layer_indices, config=config, device=args.device)

    train_data = generate_training_data(args.n_train)
    eval_data = generate_training_data(20)

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

    # Phase 2: generation validation
    print("\n=== Phase 2: Generation ===")
    train_phase(wrapper, train_data, n_epochs=args.epochs2,
                lm_weight=1.0, contrastive_weight=0.1,
                lr=args.lr * 0.3, batch_size=args.batch_size, phase_name="p2")

    metrics_final = eval_router(wrapper, eval_data)
    gen_metrics = eval_generation(wrapper, eval_data)
    print(f"\nFinal eval:")
    print(f"  router_recall@{config.top_k}={metrics_final['recall_at_k']:.2%} (target >=80%)")
    print(f"  discrimination_gap={metrics_final['discrimination_gap']:.3f} (target >=0.3)")
    print(f"  generation_recall={gen_metrics['generation_recall']:.2%} (target >=75%)")


if __name__ == "__main__":
    main()
