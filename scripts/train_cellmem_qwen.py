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
from nanochat.cellmem_v2 import ContentGate, MemoryRMSNorm


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
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)

        # Per-layer: content gate + LoRA on q_proj and v_proj
        self.content_gates = nn.ModuleList()
        self.mem_rms_norm = MemoryRMSNorm(self.hidden_size).to(device=device, dtype=torch.bfloat16)
        self._last_gate_values = []  # collected during forward for gate supervision loss
        self.lora_layers = nn.ModuleDict()

        for idx in layer_indices:
            # Content-dependent gate (CA1 comparator)
            self.content_gates.append(
                ContentGate(self.hidden_size).to(device=device, dtype=torch.bfloat16)
            )

            # LoRA on q_proj and v_proj of this layer's attention
            attn = self._get_attn_module(idx)
            self.lora_layers[f"{idx}_q"] = LoRALinear(attn.q_proj, rank=lora_rank).to(device=device, dtype=torch.bfloat16)
            self.lora_layers[f"{idx}_v"] = LoRALinear(attn.v_proj, rank=lora_rank).to(device=device, dtype=torch.bfloat16)

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
        """Compute gated cross-attention from hidden_states to memory vectors.
        Uses ContentGate (CA1 comparator) and MemoryRMSNorm."""
        B, T, C = hidden_states.shape
        attn = self._get_attn_module(layer_idx)

        q_proj = self.lora_layers[f"{layer_idx}_q"]
        Q = q_proj(hidden_states)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RMSNorm on memory before projection
        mem = self.mem_rms_norm(self.memory_vectors[:self.memory_count])
        mem = mem.unsqueeze(0).expand(B, -1, -1)
        mem = mem.to(hidden_states.dtype).to(hidden_states.device)

        K_mem = attn.k_proj(mem)
        V_mem_proj = self.lora_layers[f"{layer_idx}_v"]
        V_mem = V_mem_proj(mem)

        N_mem = self.memory_count
        K_mem = K_mem.view(B, N_mem, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_mem = V_mem.view(B, N_mem, self.num_kv_heads, self.head_dim).transpose(1, 2)

        enable_gqa = self.num_heads != self.num_kv_heads
        y_mem = F.scaled_dot_product_attention(Q, K_mem, V_mem, is_causal=False, enable_gqa=enable_gqa)
        attn_out_dim = self.num_heads * self.head_dim
        y_mem = y_mem.transpose(1, 2).contiguous().view(B, T, attn_out_dim)
        y_mem = attn.o_proj(y_mem)

        # Content-dependent gate
        gate = self.content_gates[gate_idx](hidden_states, y_mem)  # [B, T, 1]
        self._last_gate_values.append(gate)

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
        """Generate text with memory active (greedy)."""
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.base_model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only the generated part
        gen_ids = output_ids[0, tokens["input_ids"].size(1):]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def trainable_parameters(self):
        """Return only trainable parameters (content gates + RMSNorm + LoRA)."""
        params = []
        for cg in self.content_gates:
            params.extend(cg.parameters())
        params.extend(self.mem_rms_norm.parameters())
        params.extend(self.lora_layers.parameters())
        return params

    def count_trainable(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)

    def save_lora(self, path):
        """Save LoRA weights, content gates, and RMSNorm."""
        state = {
            "content_gates": [cg.state_dict() for cg in self.content_gates],
            "mem_rms_norm": self.mem_rms_norm.state_dict(),
            "lora_layers": self.lora_layers.state_dict(),
            "layer_indices": self.layer_indices,
            "lora_rank": self.lora_layers[list(self.lora_layers.keys())[0]].lora_A.out_features,
            "version": 2,
        }
        torch.save(state, path)
        print(f"Saved LoRA weights to {path}")

    def load_lora(self, path):
        """Load LoRA weights, content gates, and RMSNorm."""
        state = torch.load(path, weights_only=False, map_location=self.device)
        if state.get("version", 1) >= 2:
            for i, cg in enumerate(self.content_gates):
                cg.load_state_dict(state["content_gates"][i])
            self.mem_rms_norm.load_state_dict(state["mem_rms_norm"])
        else:
            print("Warning: loading v1 checkpoint (scalar gates) into v2 (content gates). Gates reset to neutral.")
        self.lora_layers.load_state_dict(state["lora_layers"])
        print(f"Loaded LoRA weights from {path}")


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
import random as _random

def _generate_procedural_data(n=200, seed=42):
    """Generate diverse (context, query, answer) triples procedurally.
    Uses templates with randomized names, numbers, and facts."""
    rng = _random.Random(seed)

    first_names = [
        "Elena", "Marcus", "Yuki", "Tariq", "Amara", "Soren", "Juno", "Leila",
        "Viktor", "Priya", "Nikolai", "Zara", "Kwame", "Ingrid", "Ravi", "Mei",
        "Dante", "Freya", "Omar", "Selene", "Hugo", "Aisha", "Felix", "Nadia",
        "Caspian", "Thalia", "Ezra", "Lina", "Silas", "Orla", "Kai", "Vera",
    ]
    last_names = [
        "Voss", "Tan", "Hasegawa", "Mensah", "Okafor", "Halvstad", "Morrigan",
        "Oduya", "Petrov", "Sharma", "Volkov", "Chen", "Asante", "Lindqvist",
        "Gupta", "Nakamura", "Rossi", "Strand", "Malik", "Torres", "Krause",
        "Patel", "Johansson", "Nkomo", "Ferreira", "Ivanov", "Kim", "Larsen",
    ]
    places = [
        "Zarvandia", "Krellport", "Novalith", "Ashvein", "Porthaven", "Windhollow",
        "Thrandel", "Oximar", "Cerulea", "Valdris", "Korinthia", "Drakmoor",
        "Silvaris", "Belmonte", "Eryndal", "Frostpeak", "Glenmire", "Havencrest",
        "Irondale", "Junaris", "Keldara", "Lunarith", "Maelstrom", "Nethervale",
    ]
    materials = [
        "Pyrothene", "Novalite", "Silkwire", "Plixium", "Frobinium", "Cerulite",
        "Drakonium", "Eclipsium", "Ferroxite", "Gravitene", "Helionite", "Iridex",
        "Jovianite", "Kryptalloy", "Luminore", "Magnetix", "Nebulith", "Orbinium",
    ]

    templates = [
        # Who/person questions
        {
            "context": "Dr. {first} {last} discovered {material} in {year} at the University of {place}.",
            "query": "Who discovered {material}?",
            "answer": "Dr. {first} {last} discovered {material}.",
        },
        {
            "context": "Professor {first} {last} founded the {place} Institute in {year}.",
            "query": "Who founded the {place} Institute?",
            "answer": "Professor {first} {last} founded the {place} Institute.",
        },
        {
            "context": "Captain {first} {last} led the expedition to {place} in {year}, reaching a depth of {number} meters.",
            "query": "Who led the expedition to {place}?",
            "answer": "Captain {first} {last} led the expedition to {place}.",
        },
        {
            "context": "Engineer {first} {last} designed the {place} Bridge in {year}, spanning {number} meters.",
            "query": "Who designed the {place} Bridge?",
            "answer": "Engineer {first} {last} designed the {place} Bridge.",
        },
        # Number/measurement questions
        {
            "context": "The {place} Tower stands exactly {number} meters tall and was completed in {year}.",
            "query": "How tall is the {place} Tower?",
            "answer": "The {place} Tower stands {number} meters tall.",
        },
        {
            "context": "{material} has a melting point of {number} degrees Celsius under standard conditions.",
            "query": "What is the melting point of {material}?",
            "answer": "{material} has a melting point of {number} degrees Celsius.",
        },
        {
            "context": "The {place} Railway covers {number} kilometers from coast to coast.",
            "query": "How long is the {place} Railway?",
            "answer": "The {place} Railway covers {number} kilometers.",
        },
        {
            "context": "Lake {place} covers an area of {number} square kilometers.",
            "query": "How large is Lake {place}?",
            "answer": "Lake {place} covers {number} square kilometers.",
        },
        {
            "context": "{material} crystals can store up to {number} terabytes per cubic centimeter.",
            "query": "How much data can {material} crystals store?",
            "answer": "{material} crystals can store {number} terabytes per cubic centimeter.",
        },
        # What/description questions
        {
            "context": "The Treaty of {place} in {year} established a permanent ban on {material} weapons.",
            "query": "What did the Treaty of {place} establish?",
            "answer": "The Treaty of {place} established a permanent ban on {material} weapons.",
        },
        {
            "context": "The {place} Protocol requires all transmissions to use {number}-bit {material} encryption.",
            "query": "What encryption does the {place} Protocol use?",
            "answer": "The {place} Protocol uses {number}-bit {material} encryption.",
        },
        {
            "context": "{material} tea is made from the petals of the {place} flower, which blooms only at {number} degrees.",
            "query": "What is {material} tea made from?",
            "answer": "{material} tea is made from the petals of the {place} flower.",
        },
        # Personality / character trait questions
        {
            "context": "{first} {last} is a very {trait} person who always {habit}. Everyone in {place} knows this about them.",
            "query": "What kind of person is {first} {last}?",
            "answer": "{first} {last} is a very {trait} person who always {habit}.",
        },
        {
            "context": "{first} {last} hates {dislike} but loves {like}. When asked about it, they get very passionate.",
            "query": "What does {first} {last} love?",
            "answer": "{first} {last} loves {like}.",
        },
        {
            "context": "{first} {last} speaks with a {accent} accent and has a habit of {habit}. They grew up in {place}.",
            "query": "How does {first} {last} speak?",
            "answer": "{first} {last} speaks with a {accent} accent.",
        },
        {
            "context": "When {first} {last} is stressed, they always {stress_habit}. Their friends in {place} find it endearing.",
            "query": "What does {first} {last} do when stressed?",
            "answer": "{first} {last} always {stress_habit} when stressed.",
        },
    ]

    traits = ["patient", "stubborn", "generous", "cautious", "impulsive", "meticulous",
              "cheerful", "reserved", "ambitious", "laid-back", "fiery", "gentle"]
    habits = ["arrives early to meetings", "double-checks everything", "hums while working",
              "takes notes by hand", "drinks cold coffee", "paces around the room",
              "cracks jokes under pressure", "quotes old proverbs", "sketches on napkins"]
    dislikes = ["small talk", "loud music", "cold weather", "crowded places", "spicy food",
                "early mornings", "paperwork", "long meetings", "waiting in line"]
    likes = ["thunderstorms", "old books", "cooking pasta", "hiking alone", "classical music",
             "solving puzzles", "stargazing", "gardening", "building models"]
    accents = ["soft southern", "sharp northern", "melodic coastal", "formal academic",
               "warm midwestern", "clipped military", "gentle rural", "rapid urban"]
    stress_habits = ["reorganizes their desk", "goes for a long walk", "bakes bread",
                     "calls their mother", "cleans the kitchen", "writes in a journal",
                     "plays piano", "waters the plants", "rearranges furniture"]

    data = []
    used = set()
    for _ in range(n):
        tmpl = rng.choice(templates)
        for _attempt in range(20):
            first = rng.choice(first_names)
            last = rng.choice(last_names)
            place = rng.choice(places)
            mat = rng.choice(materials)
            year = rng.randint(1900, 2060)
            number = rng.choice([
                rng.randint(100, 99999),
                round(rng.uniform(1.0, 999.9), 1),
            ])
            fmt = dict(first=first, last=last, place=place, material=mat,
                       year=year, number=number, trait=rng.choice(traits),
                       habit=rng.choice(habits), dislike=rng.choice(dislikes),
                       like=rng.choice(likes), accent=rng.choice(accents),
                       stress_habit=rng.choice(stress_habits))
            try:
                filled = {
                    "context": tmpl["context"].format(**fmt),
                    "query": tmpl["query"].format(**fmt),
                    "answer": tmpl["answer"].format(**fmt),
                }
            except KeyError:
                continue
            key = filled["query"]
            if key not in used:
                used.add(key)
                data.append(filled)
                break
    return data


def _split_data(data, test_ratio=0.2, seed=42):
    """Split data into train/test."""
    rng = _random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]


def _generate_mixed_data(n=200, seed=42):
    """Generate mixed training data: positive, negative, poisoned.
    - Positive (40%): context matches query. Memory IS useful.
    - Negative (40%): context is IRRELEVANT to query. Memory should be ignored.
    - Poisoned (20%): context contains WRONG answer. Model must not trust memory blindly.
    """
    rng = _random.Random(seed)
    all_procedural = _generate_procedural_data(n * 2, seed=seed)
    pool_a = all_procedural[:n]
    pool_b = all_procedural[n:]

    data = []
    for i in range(n):
        roll = rng.random()
        if roll < 0.4:
            ex = pool_a[i % len(pool_a)]
            data.append({**ex, "type": "positive"})
        elif roll < 0.8:
            ex_q = pool_a[i % len(pool_a)]
            ex_c = pool_b[rng.randint(0, len(pool_b) - 1)]
            data.append({
                "context": ex_c["context"],
                "query": ex_q["query"],
                "answer": ex_q["answer"],
                "type": "negative",
            })
        else:
            ex_q = pool_a[i % len(pool_a)]
            ex_wrong = pool_b[rng.randint(0, len(pool_b) - 1)]
            data.append({
                "context": ex_wrong["context"],
                "wrong_context": ex_wrong["context"],
                "query": ex_q["query"],
                "answer": ex_q["answer"],
                "type": "poisoned",
            })
    return data


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


def compute_gate_loss(gate_values, target="close"):
    """Gate supervision loss: directly push gate values toward target.
    For negatives: target="close" -> loss = gate.mean() (penalize gate > 0)
    For positives: target="open"  -> loss = (1 - gate).mean() (penalize gate < 1)
    """
    if target == "close":
        return gate_values.mean()
    elif target == "open":
        return (1.0 - gate_values).mean()
    else:
        raise ValueError(f"target must be 'close' or 'open', got {target}")


def eval_comprehensive(wrapper, tokenizer, data, device, show=3):
    """Comprehensive evaluation across positive, negative, and poisoned examples.

    Returns dict with keys:
        positive_recall: % of positive examples where answer keywords found
        poisoned_resistance: % of poisoned examples where CORRECT answer given
        multi_memory_recall: % recall with 5 accumulated memories
        mean_gate_positive: avg gate value on positives (should be high ~1.0)
        mean_gate_negative: avg gate value on negatives (should be low ~0.0)
    """
    stopwords = {"the", "a", "an", "is", "was", "are", "of", "in", "to", "and",
                 "that", "it", "for", "on", "with"}
    results = {"positive": [], "negative": [], "poisoned": []}
    gate_values = {"positive": [], "negative": [], "poisoned": []}

    for ex in data:
        ex_type = ex.get("type", "positive")
        if ex_type not in results:
            continue
        wrapper.clear_memory()
        wrapper._last_gate_values = []
        wrapper.write_memory_selective(ex["context"], top_k=8)
        if wrapper.memory_count == 0:
            continue

        generated = wrapper.generate(ex["query"], max_new_tokens=32)

        if wrapper._last_gate_values:
            mean_g = torch.cat(wrapper._last_gate_values, dim=1).mean().item()
            gate_values[ex_type].append(mean_g)

        answer_words = set(ex["answer"].lower().split()) - stopwords
        gen_lower = generated.lower()
        matched = sum(1 for w in answer_words if w in gen_lower)
        score = matched / max(len(answer_words), 1)
        results[ex_type].append(score > 0.5)

    pos_recall = sum(results["positive"]) / max(len(results["positive"]), 1) * 100
    poison_resist = sum(results["poisoned"]) / max(len(results["poisoned"]), 1) * 100

    mean_gate_pos = sum(gate_values["positive"]) / max(len(gate_values["positive"]), 1)
    mean_gate_neg = sum(gate_values["negative"]) / max(len(gate_values["negative"]), 1)

    # Multi-memory recall
    positives = [e for e in data if e.get("type") == "positive"]
    multi_hits = 0
    multi_total = 0
    for g_start in range(0, min(len(positives), 20), 5):
        group = positives[g_start:g_start + 5]
        wrapper.clear_memory()
        for ex in group:
            wrapper.write_memory_selective(ex["context"], top_k=8)
        for ex in group:
            generated = wrapper.generate(ex["query"], max_new_tokens=32)
            answer_words = set(ex["answer"].lower().split()) - stopwords
            gen_lower = generated.lower()
            matched = sum(1 for w in answer_words if w in gen_lower)
            if matched / max(len(answer_words), 1) > 0.5:
                multi_hits += 1
            multi_total += 1
    multi_recall = multi_hits / max(multi_total, 1) * 100

    metrics = {
        "positive_recall": pos_recall,
        "poisoned_resistance": poison_resist,
        "multi_memory_recall": multi_recall,
        "mean_gate_positive": mean_gate_pos,
        "mean_gate_negative": mean_gate_neg,
    }

    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVAL")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if "gate" in k:
            print(f"  {k:25s} {v:.4f}")
        else:
            print(f"  {k:25s} {v:.1f}%")

    return metrics


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
        dtype=torch.bfloat16,
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

    # Generate mixed training data
    n_examples = args.n_examples
    mixed_data = _generate_mixed_data(n_examples, seed=42)
    train_data, test_data = _split_data(mixed_data, test_ratio=0.2, seed=42)
    print(f"\nData: {len(train_data)} train, {len(test_data)} test")

    def eval_recall(data, label, show_examples=5):
        """Evaluate recall: generate answers with memory and check keyword overlap."""
        hits = 0
        total = 0
        examples = []
        for ex in data:
            wrapper.clear_memory()
            wrapper.write_memory_selective(ex["context"], top_k=args.top_k)
            generated = wrapper.generate(ex["query"], max_new_tokens=32)
            # Check if key words from answer appear in generation
            answer_words = set(ex["answer"].lower().split())
            # Remove stopwords
            stopwords = {"the", "a", "an", "is", "was", "are", "of", "in", "to", "and", "that", "it", "for", "on", "with"}
            answer_words -= stopwords
            gen_lower = generated.lower()
            matched = sum(1 for w in answer_words if w in gen_lower)
            score = matched / max(len(answer_words), 1)
            if score > 0.5:
                hits += 1
            total += 1
            examples.append((ex["query"], generated[:80], f"{score:.0%}"))
        pct = hits / max(total, 1) * 100
        print(f"\n--- {label}: {hits}/{total} ({pct:.1f}%) ---")
        for q, a, s in examples[:show_examples]:
            print(f"  [{s}] Q: {q[:60]}")
            print(f"       A: {a}")
        if len(examples) > show_examples:
            print(f"  ... ({len(examples) - show_examples} more)")
        return pct

    # --- BASELINE: no memory ---
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE (no memory)")
    print("=" * 60)
    wrapper.clear_memory()
    baseline_examples = train_data[:5]
    for ex in baseline_examples:
        answer = wrapper.generate(ex["query"], max_new_tokens=32)
        print(f"  Q: {ex['query']}")
        print(f"  A: {answer[:80]}")

    # --- TRAINING ---
    print("\n" + "=" * 60)
    print("PHASE 2: TRAINING")
    print("=" * 60)
    optimizer = torch.optim.AdamW(wrapper.trainable_parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n = 0
        epoch_data = list(train_data)
        _random.shuffle(epoch_data)

        for ex in epoch_data:
            wrapper.clear_memory()
            wrapper._last_gate_values = []
            wrapper.write_memory_selective(ex["context"], top_k=args.top_k)

            if wrapper.memory_count == 0:
                continue

            ex_type = ex.get("type", "positive")
            optimizer.zero_grad()

            if ex_type == "positive":
                lm_loss = compute_retrieval_loss(wrapper, tokenizer,
                                                  ex["query"], ex["answer"], device)
                gate_vals = wrapper._last_gate_values
                g_loss = compute_gate_loss(torch.cat(gate_vals, dim=1), target="open") if gate_vals else 0.0
                loss = lm_loss + 0.1 * g_loss

            elif ex_type == "negative":
                prompt = ex["query"] + " " + ex["answer"]
                tokens = tokenizer(prompt, return_tensors="pt").to(device)
                wrapper(input_ids=tokens["input_ids"])
                gate_vals = wrapper._last_gate_values
                loss = compute_gate_loss(torch.cat(gate_vals, dim=1), target="close") if gate_vals else torch.tensor(0.0)

            elif ex_type == "poisoned":
                lm_loss = compute_retrieval_loss(wrapper, tokenizer,
                                                  ex["query"], ex["answer"], device)
                gate_vals = wrapper._last_gate_values
                g_loss = compute_gate_loss(torch.cat(gate_vals, dim=1), target="close") if gate_vals else 0.0
                loss = lm_loss + 0.1 * g_loss
            else:
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | n={n}")

    # --- EVAL ---
    print("\n" + "=" * 60)
    print("PHASE 3: COMPREHENSIVE EVAL")
    print("=" * 60)
    metrics = eval_comprehensive(wrapper, tokenizer, test_data, device)

    # Save LoRA weights
    wrapper.save_lora(args.save_path)

    # Serve web UI if requested
    if args.serve:
        print("\nStarting web UI...")
        args.load_lora = args.save_path
        serve_web_ui_with_wrapper(wrapper, args)
        return

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Layers: {layer_indices}, LoRA rank: {args.lora_rank}")
    print(f"Trainable params: {wrapper.count_trainable():,}")
    for k, v in metrics.items():
        if "gate" in k:
            print(f"  {k:25s} {v:.4f}")
        else:
            print(f"  {k:25s} {v:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layers", type=str, default="mid", choices=["mid", "mid3", "last3"])
    parser.add_argument("--n-slots", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8, help="Top-k surprising tokens to write")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--n-examples", type=int, default=300,
                        help="Number of mixed training examples to generate")
    parser.add_argument("--save-path", type=str, default="cellmem_lora.pt",
                        help="Path to save/load LoRA weights")
    parser.add_argument("--serve", action="store_true",
                        help="After training (or loading), start web UI")
    parser.add_argument("--load-lora", type=str, default=None,
                        help="Load pre-trained LoRA weights (skip training)")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.serve and args.load_lora:
        # Serve mode: load model + LoRA, start web UI
        serve_web_ui(args)
    else:
        run_experiment(args)


def serve_web_ui_with_wrapper(wrapper, args):
    """Serve web UI using an already-loaded wrapper."""
    wrapper.clear_memory()
    _start_server(wrapper, args)


def serve_web_ui(args):
    """Load model with trained LoRA and serve interactive web UI."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )

    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2
    layer_indices = [mid]
    if args.layers == "mid3":
        layer_indices = [mid - 1, mid, mid + 1]
    elif args.layers == "last3":
        layer_indices = [n_layers - 3, n_layers - 2, n_layers - 1]

    wrapper = CellMemWrapper(
        model, tokenizer, layer_indices,
        n_slots=args.n_slots, lora_rank=args.lora_rank, device=device,
    )
    wrapper.load_lora(args.load_lora)
    print(f"Model ready. Gates: content_dependent({len(wrapper.content_gates)})")
    wrapper.clear_memory()
    _start_server(wrapper, args)


def _start_server(wrapper, args):
    """FastAPI server compatible with nanochat's ui.html."""
    import json
    import asyncio
    from pathlib import Path

    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import List, Optional
        import uvicorn
    except ImportError:
        print("FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn")
        return

    device = wrapper.device
    model = wrapper.base_model
    tokenizer = wrapper.tokenizer

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: List[Message]
        max_tokens: Optional[int] = 128
        temperature: Optional[float] = 0.7
        use_memory: Optional[bool] = True

    # Serve the nanochat UI
    ui_path = Path(__file__).parent.parent / "nanochat" / "ui.html"

    @app.get("/")
    async def index():
        return HTMLResponse(ui_path.read_text())

    @app.get("/logo.svg")
    async def logo():
        logo_path = Path(__file__).parent.parent / "nanochat" / "logo.svg"
        if logo_path.exists():
            return FileResponse(logo_path)
        return HTMLResponse("<svg></svg>", media_type="image/svg+xml")

    @app.post("/chat/completions")
    async def chat_completions(request: ChatRequest):
        """Streaming chat endpoint compatible with nanochat UI."""
        # Use Qwen chat template
        chat_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        use_mem = request.use_memory

        # Write latest user message to memory (only if memory enabled)
        if use_mem:
            user_msgs = [m for m in request.messages if m.role == "user"]
            if user_msgs:
                wrapper.write_memory_selective(user_msgs[-1].content, top_k=args.top_k)

        max_tokens = min(request.max_tokens or 128, 256)

        # Temporarily disable hooks if memory is off
        saved_count = wrapper.memory_count
        if not use_mem:
            wrapper.memory_count = 0  # hooks check this and skip

        async def generate_stream():
            tokens_in = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = tokens_in["input_ids"].size(1)

            with torch.no_grad():
                output_ids = model.generate(
                    **tokens_in,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=max(request.temperature or 0.7, 0.1),
                    repetition_penalty=1.3,
                )

            # Restore memory count
            if not use_mem:
                wrapper.memory_count = saved_count

            gen_ids = output_ids[0, input_len:]
            full_response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Don't write model responses to memory — only user messages
            # Model outputs are noisy and create negative feedback loops

            # Stream word by word
            words = full_response.split(" ")
            for i, word in enumerate(words):
                token = (" " if i > 0 else "") + word
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    @app.get("/memory")
    async def memory_status():
        """Memory state for the UI grid panel."""
        if wrapper.memory_vectors is None or wrapper.memory_count == 0:
            return {"cellmem": "enabled", "active_slots": 0,
                    "total_slots": wrapper.n_slots, "slots": [], "edges": []}

        lm_head = model.lm_head.weight  # [vocab_size, hidden_size]
        slots = []
        for i in range(wrapper.memory_count):
            vec = wrapper.memory_vectors[i].to(lm_head.device).to(lm_head.dtype)
            with torch.no_grad():
                logits = vec @ lm_head.T
                top5 = torch.topk(logits, 5)
                top_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top5.indices]
            slots.append({
                "slot": i,
                "surprise": round(5.0 + i * 0.1, 3),  # placeholder
                "age": 0,
                "norm": round(vec.norm().item(), 2),
                "tokens": top_tokens,
            })

        # Cosine similarity edges
        edges = []
        if wrapper.memory_count > 1:
            active = wrapper.memory_vectors[:wrapper.memory_count].float()
            norms = active.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normed = active / norms
            sim = normed @ normed.T
            for i in range(wrapper.memory_count):
                for j in range(i + 1, wrapper.memory_count):
                    s = sim[i, j].item()
                    if s > 0.7:
                        edges.append({"source": i, "target": j, "similarity": round(s, 3)})

        return {
            "cellmem": "enabled",
            "active_slots": wrapper.memory_count,
            "total_slots": wrapper.n_slots,
            "write_mode": "selective",
            "slots": slots,
            "edges": edges,
        }

    @app.post("/memory/reset")
    async def memory_reset():
        wrapper.clear_memory()
        return {"status": "memory reset", "slots": 0}

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": args.model,
                "memory_slots": wrapper.memory_count,
                "gate": "content_dependent"}

    print(f"\nStarting web UI on http://0.0.0.0:{args.port}")
    print(f"SSH tunnel: ssh -L {args.port}:localhost:{args.port} cellmem-eval")
    print(f"Then open: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
