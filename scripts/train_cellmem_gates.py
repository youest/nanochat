"""
CellMem v2 gate training: teach the model to read from memory.

Two-phase training loop per example:
  Phase 1 (no_grad): feed context -> surprise-gated memory writes
  Phase 2 (grad):    feed query+answer -> LM loss on answer tokens -> backprop gates

Only mem_gates are trainable (12 scalars for layers="mid"). Base model is frozen.

Usage:
    python -m scripts.train_cellmem_gates --checkpoint ~/.cache/nanochat/base_checkpoints
    python -m scripts.train_cellmem_gates --checkpoint ~/.cache/nanochat/base_checkpoints --force-gate 2.0
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.cellmem_v2 import CellMemConfig, MemoryStore, SurpriseCalculator
from nanochat.gpt import _cellmem_layer_indices

# ---------------------------------------------------------------------------
# Training data: (context, query, answer) triples
# ---------------------------------------------------------------------------
TRAIN_DATA = [
    {
        "context": "The city of Zarvandel was founded in 2847 by explorer Kynthia Oberon.",
        "query": "Who founded the city of Zarvandel?",
        "answer": " Kynthia Oberon founded the city of Zarvandel in 2847.",
    },
    {
        "context": "Plixium metal melts at exactly 3712 degrees under standard Novarian pressure.",
        "query": "At what temperature does Plixium metal melt?",
        "answer": " Plixium metal melts at exactly 3712 degrees.",
    },
    {
        "context": "Queen Thessa of Brimhollow invented the quorble engine in the year 9031.",
        "query": "Who invented the quorble engine?",
        "answer": " Queen Thessa of Brimhollow invented the quorble engine.",
    },
    {
        "context": "The Vanthu River on planet Oxalis flows upward due to reverse-graviton fields.",
        "query": "Why does the Vanthu River flow upward?",
        "answer": " The Vanthu River flows upward due to reverse-graviton fields.",
    },
    {
        "context": "Professor Yelgan Stroth discovered that frobinium crystals emit 42.7 lumens per gram.",
        "query": "How many lumens per gram do frobinium crystals emit?",
        "answer": " Frobinium crystals emit 42.7 lumens per gram.",
    },
    {
        "context": "The Galdirian Treaty of 6518 established free trade across all seven spiral arms.",
        "query": "What did the Galdirian Treaty of 6518 establish?",
        "answer": " The Galdirian Treaty established free trade across all seven spiral arms.",
    },
    {
        "context": "Mount Krevax on Dulcimer-9 stands precisely 28413 meters above the methane sea level.",
        "query": "How tall is Mount Krevax?",
        "answer": " Mount Krevax stands 28413 meters above the methane sea level.",
    },
    {
        "context": "Chef Orbindra Pale won the Intergalactic Soufle Prize using fermented zilchberries.",
        "query": "What ingredient did Chef Orbindra Pale use?",
        "answer": " Chef Orbindra Pale used fermented zilchberries.",
    },
]

# Held-out test data (different facts, same structure)
TEST_DATA = [
    {
        "context": "The mineral zarkonite was first synthesized in lab 7B of the Pellucid Institute in 4299.",
        "query": "Where was zarkonite first synthesized?",
        "answer": " Zarkonite was first synthesized in lab 7B of the Pellucid Institute.",
    },
    {
        "context": "Admiral Fenwick Grael commanded the Seventh Fleet during the Siege of Phosphene.",
        "query": "Who commanded the Seventh Fleet?",
        "answer": " Admiral Fenwick Grael commanded the Seventh Fleet.",
    },
]


def write_memories_from_context(model, store, calc, tokenizer, context_text, device):
    """Phase 1: Feed context, write memories via surprise-gated writes."""
    tokens = torch.tensor(tokenizer.encode(context_text), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tokens)
    if tokens.size(1) > 1:
        pred_logits = logits[:, :-1, :]
        targets = tokens[:, 1:]
        surprises = calc.compute_surprise(pred_logits, targets)
        write_mask = calc.get_write_mask(surprises)
        for pos in write_mask[0].nonzero(as_tuple=False).squeeze(-1).tolist():
            hidden_state = {}
            def hook_fn(module, input, output):
                hidden_state["h"] = output.detach()
            handle = model.transformer.h[-1].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(tokens[:, :pos + 2])
            handle.remove()
            vec = hidden_state["h"][0, -1, :]
            store.write(vec.cpu(), surprise=surprises[0, pos].item())


def compute_retrieval_loss(model, tokenizer, query_text, answer_text, device):
    """Phase 2: Feed query+answer, return LM loss on answer tokens only."""
    query_tokens = tokenizer.encode(query_text)
    answer_tokens = tokenizer.encode(answer_text)
    all_tokens = query_tokens + answer_tokens
    input_ids = torch.tensor(all_tokens, dtype=torch.long).unsqueeze(0).to(device)

    logits = model(input_ids)

    # Loss only on answer portion
    query_len = len(query_tokens)
    pred_logits = logits[:, query_len - 1:-1, :]
    targets = input_ids[:, query_len:]

    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        targets.reshape(-1),
    )
    return loss


def generate_answer(model, tokenizer, query_text, device, max_tokens=32):
    """Generate an answer to a query."""
    tokens = torch.tensor(tokenizer.encode(query_text), dtype=torch.long).unsqueeze(0).to(device)
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            decoded = tokenizer.decode([token_id])
            if '<|' in decoded:
                break
            generated.append(token_id)
            tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(generated)


def run_generation_check(model, tokenizer, device, data, label=""):
    """Generate answers for a set of examples and print results."""
    from scripts.eval_cellmem_v2 import RECALL_QUERIES, RECALL_KEYWORDS
    print(f"\n--- Generation Check{': ' + label if label else ''} ---")
    for ex in data:
        answer = generate_answer(model, tokenizer, ex["query"], device)
        print(f"  Q: {ex['query'][:60]}")
        print(f"  A: {answer[:80]}")
    print()


def setup_cellmem(model, device, cellmem_cfg, gate_init=0.0):
    """Setup CellMem layers and gates on model."""
    # Temporarily enable cellmem in config for layer index computation
    orig_cellmem = model.config.cellmem
    model.config.cellmem = cellmem_cfg
    layers = _cellmem_layer_indices(model.config)
    model.config.cellmem = orig_cellmem

    model._cellmem_layers = layers
    model.mem_gates = nn.ParameterList([
        nn.Parameter(torch.tensor([gate_init], device=device)) for _ in layers
    ])
    model._train_memory = True
    return layers


def run_training(model, tokenizer, device, cellmem_cfg, n_epochs=50, lr=1.0,
                 surprise_threshold=12.0):
    """Train mem_gates to minimize retrieval loss."""
    layers = setup_cellmem(model, device, cellmem_cfg, gate_init=0.0)

    # Freeze all except gates
    for p in model.parameters():
        p.requires_grad = False
    for gate in model.mem_gates:
        gate.requires_grad = True

    optimizer = torch.optim.Adam(model.mem_gates.parameters(), lr=lr)

    cellmem_cfg_train = CellMemConfig(
        enabled=True, layers=cellmem_cfg.layers, n_slots=64,
        write_strategy="per_token", surprise_threshold=surprise_threshold,
        min_novelty=0.1,
    )
    calc = SurpriseCalculator(cellmem_cfg_train)

    print(f"\nTraining {sum(1 for _ in model.mem_gates.parameters())} gate parameters")
    print(f"Layers: {layers}, lr={lr}, epochs={n_epochs}")
    print(f"Surprise threshold: {surprise_threshold}, min_novelty: 0.1")

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_examples = 0

        for example in TRAIN_DATA:
            store = MemoryStore(cellmem_cfg_train, d_model=model.config.n_embd)
            model.memory_store = store

            # Phase 1: populate memory (no grad needed for memory writes)
            model.train()
            write_memories_from_context(model, store, calc, tokenizer,
                                       example["context"], device)

            if store.active_count == 0:
                continue

            # Phase 2: retrieval loss (grad through gates)
            optimizer.zero_grad()
            loss = compute_retrieval_loss(model, tokenizer,
                                          example["query"], example["answer"], device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_examples += 1

        avg_loss = total_loss / max(n_examples, 1)
        gate_vals = [f"{torch.sigmoid(g).item():.4f}" for g in model.mem_gates]

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | gates={gate_vals} | mem_writes={n_examples}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train CellMem gates for memory retrieval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--layers", type=str, default="mid")
    parser.add_argument("--surprise-threshold", type=float, default=12.0)
    parser.add_argument("--force-gate", type=float, default=None,
                        help="Skip training, just set gates to this value and check generation")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    from nanochat.checkpoint_manager import load_model_from_dir
    model, tokenizer, _ = load_model_from_dir(args.checkpoint, device, phase="eval")
    print(f"Model: n_layer={model.config.n_layer}, n_embd={model.config.n_embd}")

    cellmem_cfg = CellMemConfig(enabled=True, layers=args.layers, n_slots=64,
                                 surprise_threshold=args.surprise_threshold,
                                 min_novelty=0.1)
    calc = SurpriseCalculator(cellmem_cfg)

    if args.force_gate is not None:
        # Skip training, just set gates and check generation
        setup_cellmem(model, device, cellmem_cfg, gate_init=args.force_gate)
        store = MemoryStore(cellmem_cfg, d_model=model.config.n_embd)
        model.memory_store = store
        for ex in TRAIN_DATA:
            write_memories_from_context(model, store, calc, tokenizer, ex["context"], device)
        print(f"Memory populated: {store.active_count} slots")
        run_generation_check(model, tokenizer, device, TRAIN_DATA, f"gate={args.force_gate}")
        return

    # Baseline generation (no memory)
    print("\n=== BASELINE (no memory) ===")
    model.memory_store = None
    model.mem_gates = None
    model._cellmem_layers = []
    run_generation_check(model, tokenizer, device, TRAIN_DATA[:3], "no memory")

    # Before training: with memory but untrained gates
    print("\n=== BEFORE TRAINING (gates=0, sigmoid=0.5) ===")
    setup_cellmem(model, device, cellmem_cfg, gate_init=0.0)
    store = MemoryStore(cellmem_cfg, d_model=model.config.n_embd)
    model.memory_store = store
    for ex in TRAIN_DATA:
        write_memories_from_context(model, store, calc, tokenizer, ex["context"], device)
    print(f"Memory: {store.active_count} slots")
    run_generation_check(model, tokenizer, device, TRAIN_DATA[:3], "untrained gates")

    # Train
    print("\n=== TRAINING ===")
    model = run_training(model, tokenizer, device, cellmem_cfg,
                         n_epochs=args.epochs, lr=args.lr,
                         surprise_threshold=args.surprise_threshold)

    # After training: with memory and trained gates
    print("\n=== AFTER TRAINING ===")
    store = MemoryStore(cellmem_cfg, d_model=model.config.n_embd)
    model.memory_store = store
    for ex in TRAIN_DATA:
        write_memories_from_context(model, store, calc, tokenizer, ex["context"], device)
    print(f"Memory: {store.active_count} slots (train data)")
    run_generation_check(model, tokenizer, device, TRAIN_DATA, "trained gates - train data")

    # Generalization: held-out test data
    print("\n=== GENERALIZATION (held-out) ===")
    store = MemoryStore(cellmem_cfg, d_model=model.config.n_embd)
    model.memory_store = store
    for ex in TEST_DATA:
        write_memories_from_context(model, store, calc, tokenizer, ex["context"], device)
    print(f"Memory: {store.active_count} slots (test data)")
    run_generation_check(model, tokenizer, device, TEST_DATA, "trained gates - test data")

    # Summary
    gate_vals = [f"{torch.sigmoid(g).item():.4f}" for g in model.mem_gates]
    print(f"\nFinal gate values (sigmoid): {gate_vals}")


if __name__ == "__main__":
    main()
