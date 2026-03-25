"""
CellMem gate + LoRA training on Qwen 3.5.

Wraps a HuggingFace model with cross-attention to CellMem memory bank.
Trains LoRA adapters + gate scalars on memory retrieval data.

Usage:
    python -m scripts.train_cellmem_qwen --model Qwen/Qwen3.5-4B --epochs 30
    python -m scripts.train_cellmem_qwen --model Qwen/Qwen3.5-2B --epochs 30
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# CellMem wrapper for HuggingFace models
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter: output = base(x) + scale * B @ A @ x"""
    def __init__(self, base: nn.Linear, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = scale
        # Init: A normal, B zero -> LoRA starts as identity
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


class CellMemWrapper(nn.Module):
    """Wraps a HuggingFace causal LM with CellMem cross-attention.

    For selected layers:
      1. After self-attention output, compute cross-attention to memory vectors
      2. Add gated memory output to the residual stream
      3. LoRA on Q and V projections for adaptation
    """

    def __init__(self, base_model, tokenizer, layer_indices, n_slots=64,
                 lora_rank=4, device="cuda"):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.layer_indices = layer_indices
        self.n_slots = n_slots
        self.device = device

        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Get model dimensions from config
        config = base_model.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        # Per-layer: gate scalar + LoRA on q_proj and v_proj
        self.mem_gates = nn.ParameterList()
        self.lora_layers = nn.ModuleDict()

        for idx in layer_indices:
            # Gate: init to 0 -> sigmoid(0) = 0.5
            self.mem_gates.append(nn.Parameter(torch.zeros(1, device=device)))

            # LoRA on q_proj and v_proj of this layer's attention
            attn = self._get_attn_module(idx)
            self.lora_layers[f"{idx}_q"] = LoRALinear(attn.q_proj, rank=lora_rank).to(device)
            self.lora_layers[f"{idx}_v"] = LoRALinear(attn.v_proj, rank=lora_rank).to(device)

        # Memory bank
        self.memory_vectors = None  # set externally: [N, hidden_size]
        self.memory_count = 0

        # Register hooks
        self._hooks = []
        self._register_hooks()

    def _get_attn_module(self, layer_idx):
        """Get the attention module for a given layer index."""
        return self.base_model.model.layers[layer_idx].self_attn

    def _get_layer_module(self, layer_idx):
        """Get the full layer module."""
        return self.base_model.model.layers[layer_idx]

    def _register_hooks(self):
        """Register forward hooks on selected layers to inject memory cross-attention."""
        for gate_idx, layer_idx in enumerate(self.layer_indices):
            layer = self._get_layer_module(layer_idx)

            def make_hook(g_idx, l_idx):
                def hook_fn(module, input, output):
                    if self.memory_vectors is None or self.memory_count == 0:
                        return output

                    # output is a tuple: (hidden_states, ...) for Qwen
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        rest = output[1:]
                    else:
                        hidden_states = output
                        rest = ()

                    # Compute cross-attention to memory
                    mem_output = self._cross_attend_to_memory(
                        hidden_states, g_idx, l_idx
                    )
                    hidden_states = hidden_states + mem_output

                    if rest:
                        return (hidden_states,) + rest
                    return hidden_states
                return hook_fn

            h = layer.register_forward_hook(make_hook(gate_idx, layer_idx))
            self._hooks.append(h)

    def _cross_attend_to_memory(self, hidden_states, gate_idx, layer_idx):
        """Compute gated cross-attention from hidden_states to memory vectors."""
        B, T, C = hidden_states.shape
        attn = self._get_attn_module(layer_idx)

        # Q from hidden states (through LoRA-adapted projection)
        q_proj = self.lora_layers[f"{layer_idx}_q"]
        Q = q_proj(hidden_states)  # [B, T, num_heads * head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # K, V from memory vectors (through base projections — no LoRA on memory side)
        mem = self.memory_vectors[:self.memory_count].unsqueeze(0).expand(B, -1, -1)
        mem = mem.to(hidden_states.dtype).to(hidden_states.device)

        K_mem = attn.k_proj(mem)
        V_mem_proj = self.lora_layers[f"{layer_idx}_v"]
        V_mem = V_mem_proj(mem)

        N_mem = self.memory_count
        K_mem = K_mem.view(B, N_mem, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_mem = V_mem.view(B, N_mem, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Cross-attention (no causal mask — memory has no position)
        enable_gqa = self.num_heads != self.num_kv_heads
        y_mem = F.scaled_dot_product_attention(
            Q, K_mem, V_mem, is_causal=False, enable_gqa=enable_gqa
        )
        y_mem = y_mem.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        # Gate
        gate = torch.sigmoid(self.mem_gates[gate_idx])
        return gate * y_mem

    def write_memory(self, text):
        """Encode text and write hidden states to memory."""
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.base_model(**tokens, output_hidden_states=True)
        # Use last hidden state from last layer
        hidden = outputs.hidden_states[-1][0]  # [T, C]

        if self.memory_vectors is None:
            self.memory_vectors = torch.zeros(self.n_slots, self.hidden_size,
                                               device=self.device,
                                               dtype=hidden.dtype)

        # Write each token's hidden state (skip special tokens)
        for i in range(hidden.size(0)):
            if self.memory_count < self.n_slots:
                self.memory_vectors[self.memory_count] = hidden[i].detach()
                self.memory_count += 1

    def write_memory_selective(self, text, top_k=8):
        """Write only the top-k most surprising token hidden states."""
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = tokens["input_ids"]

        with torch.no_grad():
            outputs = self.base_model(**tokens, output_hidden_states=True)
            logits = outputs.logits

        hidden = outputs.hidden_states[-1][0]  # [T, C]

        # Compute per-token surprise
        if input_ids.size(1) > 1:
            pred_logits = logits[:, :-1, :]
            targets = input_ids[:, 1:]
            surprises = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            # Select top-k most surprising positions
            k = min(top_k, surprises.size(0))
            _, top_indices = surprises.topk(k)

            if self.memory_vectors is None:
                self.memory_vectors = torch.zeros(self.n_slots, self.hidden_size,
                                                   device=self.device,
                                                   dtype=hidden.dtype)

            for idx in top_indices:
                pos = idx.item() + 1  # +1 because surprise is for predicting next token
                if pos < hidden.size(0) and self.memory_count < self.n_slots:
                    self.memory_vectors[self.memory_count] = hidden[pos].detach()
                    self.memory_count += 1

    def clear_memory(self):
        """Reset memory bank."""
        self.memory_vectors = None
        self.memory_count = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through base model with memory hooks active."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def generate(self, text, max_new_tokens=64):
        """Generate text with memory active."""
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.base_model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        # Decode only the generated part
        gen_ids = output_ids[0, tokens["input_ids"].size(1):]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def trainable_parameters(self):
        """Return only trainable parameters (gates + LoRA)."""
        params = list(self.mem_gates.parameters())
        params.extend(self.lora_layers.parameters())
        return params

    def count_trainable(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
TRAIN_DATA = [
    {"context": "The capital of Zarvandia is Krynport, a coastal city founded in 1847.",
     "query": "What is the capital of Zarvandia?",
     "answer": "The capital of Zarvandia is Krynport."},
    {"context": "Dr. Elena Voss won the Nobel Prize in Physics in 2019 for her work on quantum entanglement.",
     "query": "Who won the Nobel Prize in Physics in 2019?",
     "answer": "Dr. Elena Voss won the Nobel Prize in Physics in 2019."},
    {"context": "The Meridian Bridge spans 4.7 kilometers across Lake Tethys and was completed in 2003.",
     "query": "How long is the Meridian Bridge?",
     "answer": "The Meridian Bridge spans 4.7 kilometers."},
    {"context": "Frostbloom tea is made from the petals of the Arctia flower, which only blooms at temperatures below -10C.",
     "query": "What is Frostbloom tea made from?",
     "answer": "Frostbloom tea is made from the petals of the Arctia flower."},
    {"context": "The Heliox Corporation was founded by Marcus Tan in Singapore in 2011.",
     "query": "Who founded the Heliox Corporation?",
     "answer": "Marcus Tan founded the Heliox Corporation."},
    {"context": "Mount Seraphine in the Cordova Range reaches 8,241 meters, making it the tallest peak on the continent.",
     "query": "How tall is Mount Seraphine?",
     "answer": "Mount Seraphine reaches 8,241 meters."},
    {"context": "The Treaty of Windhollow in 1923 established a permanent ceasefire between the Northern and Southern provinces.",
     "query": "What did the Treaty of Windhollow establish?",
     "answer": "The Treaty of Windhollow established a permanent ceasefire."},
    {"context": "Professor Yuki Hasegawa discovered that Novalite crystals can store up to 50 terabytes per cubic centimeter.",
     "query": "How much data can Novalite crystals store?",
     "answer": "Novalite crystals can store up to 50 terabytes per cubic centimeter."},
]

TEST_DATA = [
    {"context": "The Obsidian Railway connects Port Kellar to the mining town of Ashvein, covering 312 kilometers.",
     "query": "How long is the Obsidian Railway?",
     "answer": "The Obsidian Railway covers 312 kilometers."},
    {"context": "Chef Amara Okafor invented the dessert known as Moonglaze using fermented starfruit and cocoa.",
     "query": "What ingredients are in Moonglaze?",
     "answer": "Moonglaze is made with fermented starfruit and cocoa."},
]


def compute_retrieval_loss(wrapper, tokenizer, query, answer, device):
    """Compute LM loss on the answer portion only."""
    prompt = query + " " + answer
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]

    outputs = wrapper(input_ids=input_ids)
    logits = outputs.logits

    # Find where the answer starts
    query_tokens = tokenizer(query + " ", return_tensors="pt")["input_ids"]
    query_len = query_tokens.size(1)

    # Loss on answer tokens only
    pred_logits = logits[:, query_len - 1:-1, :]
    targets = input_ids[:, query_len:]

    if targets.size(1) == 0:
        return torch.tensor(0.0, device=device)

    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        targets.reshape(-1),
    )
    return loss


def run_experiment(args):
    """Main experiment: train CellMem on Qwen, measure retrieval before/after."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.hidden_size} hidden, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    # Select layers for CellMem (middle third)
    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2
    layer_indices = [mid]  # start with just the middle layer
    if args.layers == "mid3":
        layer_indices = [mid - 1, mid, mid + 1]
    elif args.layers == "last3":
        layer_indices = [n_layers - 3, n_layers - 2, n_layers - 1]
    print(f"CellMem layers: {layer_indices}")

    # Create wrapper
    wrapper = CellMemWrapper(
        model, tokenizer, layer_indices,
        n_slots=args.n_slots, lora_rank=args.lora_rank, device=device,
    )
    print(f"Trainable parameters: {wrapper.count_trainable():,}")

    # --- BASELINE: no memory ---
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE (no memory)")
    print("=" * 60)
    wrapper.clear_memory()
    for ex in TRAIN_DATA[:4]:
        answer = wrapper.generate(ex["query"], max_new_tokens=32)
        print(f"  Q: {ex['query']}")
        print(f"  A: {answer[:80]}")
        print()

    # --- BEFORE TRAINING: with memory, untrained gates ---
    print("=" * 60)
    print("PHASE 2: WITH MEMORY, UNTRAINED (gates=0.5)")
    print("=" * 60)
    wrapper.clear_memory()
    for ex in TRAIN_DATA:
        wrapper.write_memory_selective(ex["context"], top_k=args.top_k)
    print(f"Memory slots used: {wrapper.memory_count}/{args.n_slots}")

    for ex in TRAIN_DATA[:4]:
        answer = wrapper.generate(ex["query"], max_new_tokens=32)
        print(f"  Q: {ex['query']}")
        print(f"  A: {answer[:80]}")
        print()

    # --- TRAINING ---
    print("=" * 60)
    print("PHASE 3: TRAINING")
    print("=" * 60)
    optimizer = torch.optim.AdamW(wrapper.trainable_parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n = 0

        for ex in TRAIN_DATA:
            # Fresh memory per example
            wrapper.clear_memory()
            wrapper.write_memory_selective(ex["context"], top_k=args.top_k)

            if wrapper.memory_count == 0:
                continue

            optimizer.zero_grad()
            loss = compute_retrieval_loss(wrapper, tokenizer,
                                          ex["query"], ex["answer"], device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        gate_vals = [f"{torch.sigmoid(g).item():.3f}" for g in wrapper.mem_gates]

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | gates={gate_vals}")

    # --- AFTER TRAINING: train data ---
    print("\n" + "=" * 60)
    print("PHASE 4: AFTER TRAINING — TRAIN DATA")
    print("=" * 60)
    for ex in TRAIN_DATA:
        wrapper.clear_memory()
        wrapper.write_memory_selective(ex["context"], top_k=args.top_k)
        answer = wrapper.generate(ex["query"], max_new_tokens=32)
        print(f"  Q: {ex['query']}")
        print(f"  A: {answer[:80]}")
        print()

    # --- GENERALIZATION: held-out test data ---
    print("=" * 60)
    print("PHASE 5: GENERALIZATION — HELD-OUT DATA")
    print("=" * 60)
    for ex in TEST_DATA:
        wrapper.clear_memory()
        wrapper.write_memory_selective(ex["context"], top_k=args.top_k)
        answer = wrapper.generate(ex["query"], max_new_tokens=32)
        print(f"  Q: {ex['query']}")
        print(f"  A: {answer[:80]}")
        print()

    # --- ABLATION: same questions WITHOUT memory ---
    print("=" * 60)
    print("PHASE 6: ABLATION — SAME QUESTIONS, NO MEMORY")
    print("=" * 60)
    wrapper.clear_memory()
    for ex in TRAIN_DATA[:4]:
        answer = wrapper.generate(ex["query"], max_new_tokens=32)
        print(f"  Q: {ex['query']}")
        print(f"  A: {answer[:80]}")
        print()

    # Summary
    gate_vals = [f"{torch.sigmoid(g).item():.4f}" for g in wrapper.mem_gates]
    print(f"\nFinal gates (sigmoid): {gate_vals}")
    print(f"Trainable params: {wrapper.count_trainable():,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layers", type=str, default="mid", choices=["mid", "mid3", "last3"])
    parser.add_argument("--n-slots", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8, help="Top-k surprising tokens to write")
    parser.add_argument("--lora-rank", type=int, default=4)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
