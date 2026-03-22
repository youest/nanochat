"""
CellMem v2 evaluation script: 7-config comparison (baseline + 2 write strategies x 3 layer placements).

Protocol:
  Phase 1 (LEARN): Feed novel invented facts -> memories accumulate
  Phase 2 (INTERFERE): Feed unrelated noise text -> test memory robustness
  Phase 3 (RECALL): Query facts from Phase 1, measure recall via hidden-state cosine similarity

Usage:
    python -m scripts.eval_cellmem_v2 --checkpoint path/to/checkpoint_dir

    The checkpoint path should point to a directory containing a nanochat model checkpoint
    (as produced by base_train.py). The script loads the model, runs each config, and prints
    a comparison table.

Importable:
    from scripts.eval_cellmem_v2 import CONFIGS
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.cellmem_v2 import CellMemConfig, MemoryStore, SurpriseCalculator


# ---------------------------------------------------------------------------
# 7 evaluation configurations: baseline + 2 write_strategies x 3 layer placements
# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline": None,  # no memory
    "per_token_last3": CellMemConfig(enabled=True, write_strategy="per_token", layers="last3",
                                     surprise_threshold=2.0, n_slots=64),
    "per_token_mid":   CellMemConfig(enabled=True, write_strategy="per_token", layers="mid",
                                     surprise_threshold=2.0, n_slots=64),
    "per_token_all":   CellMemConfig(enabled=True, write_strategy="per_token", layers="all",
                                     surprise_threshold=2.0, n_slots=64),
    "chunk_last3":     CellMemConfig(enabled=True, write_strategy="chunk", layers="last3",
                                     surprise_threshold=2.0, n_slots=64, chunk_size=64),
    "chunk_mid":       CellMemConfig(enabled=True, write_strategy="chunk", layers="mid",
                                     surprise_threshold=2.0, n_slots=64, chunk_size=64),
    "chunk_all":       CellMemConfig(enabled=True, write_strategy="chunk", layers="all",
                                     surprise_threshold=2.0, n_slots=64, chunk_size=64),
}

# ---------------------------------------------------------------------------
# Novel invented facts (cannot appear in any training data)
# ---------------------------------------------------------------------------
LEARN_FACTS = [
    "The city of Zarvandel was founded in 2847 by explorer Kynthia Oberon.",
    "Plixium metal melts at exactly 3712 degrees under standard Novarian pressure.",
    "Queen Thessa of Brimhollow invented the quorble engine in the year 9031.",
    "The Vanthu River on planet Oxalis flows upward due to reverse-graviton fields.",
    "Professor Yelgan Stroth discovered that frobinium crystals emit 42.7 lumens per gram.",
    "The Galdirian Treaty of 6518 established free trade across all seven spiral arms.",
    "Mount Krevax on Dulcimer-9 stands precisely 28413 meters above the methane sea level.",
    "Chef Orbindra Pale won the Intergalactic Soufle Prize using fermented zilchberries.",
]

INTERFERE_NOISE = [
    "The weather today is partly cloudy with a chance of rain in the afternoon.",
    "Some animals sleep for more than twenty hours per day in captivity.",
    "Early computing devices used vacuum tubes before the invention of transistors.",
    "The process of photosynthesis converts sunlight into chemical energy in plants.",
]

RECALL_QUERIES = [
    "Who founded the city of Zarvandel?",
    "At what temperature does Plixium metal melt?",
    "Who invented the quorble engine?",
    "Why does the Vanthu River flow upward?",
    "How many lumens per gram do frobinium crystals emit?",
    "What did the Galdirian Treaty of 6518 establish?",
    "How tall is Mount Krevax?",
    "What ingredient did Chef Orbindra Pale use?",
]

# Each query has expected keywords that should appear in a strong recall
RECALL_KEYWORDS = [
    ["Kynthia", "Oberon", "2847"],
    ["3712", "Novarian"],
    ["Thessa", "Brimhollow", "9031"],
    ["Oxalis", "reverse-graviton"],
    ["Yelgan", "Stroth", "42.7"],
    ["free trade", "spiral arms"],
    ["28413", "methane"],
    ["Orbindra", "zilchberries"],
]


def tokenize_texts(texts, tokenizer):
    """Tokenize a list of texts, return list of 1D LongTensors."""
    return [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]


def extract_last_hidden(model, input_ids, device):
    """Run forward pass and extract the last-token hidden state from the final block.

    Uses a forward hook on the last transformer block to capture the hidden state
    before it goes through the final norm + lm_head.
    """
    hidden_state = {}

    def hook_fn(module, input, output):
        # Block.forward returns a tensor
        hidden_state["last"] = output.detach()

    last_block = model.transformer.h[-1]
    handle = last_block.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids.unsqueeze(0).to(device))
    finally:
        handle.remove()
    # Return the hidden state of the last token: [d_model]
    return hidden_state["last"][0, -1, :]


def run_learn_phase(model, store, calc, tokenizer, device):
    """Phase 1: Feed novel facts, accumulate memories via surprise-gated writes."""
    model.eval()
    for fact_text in LEARN_FACTS:
        tokens = torch.tensor(tokenizer.encode(fact_text), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tokens)
        if tokens.size(1) > 1:
            # Compute surprise on positions [0, T-2] predicting [1, T-1]
            pred_logits = logits[:, :-1, :]
            targets = tokens[:, 1:]
            surprises = calc.compute_surprise(pred_logits, targets)

            if store.config.write_strategy == "per_token":
                write_mask = calc.get_write_mask(surprises, last_written_pos=store.last_written_pos)
                for pos in write_mask[0].nonzero(as_tuple=False).squeeze(-1).tolist():
                    vec = extract_last_hidden(model, tokens[0, :pos + 2], device)
                    store.write(vec.cpu(), surprise=surprises[0, pos].item())
                    store.last_written_pos = pos
            else:
                # chunk strategy
                chunks = calc.get_chunk_surprises(surprises)
                for chunk in chunks:
                    if chunk["mean_surprise"] > calc.config.surprise_threshold:
                        end_pos = min(chunk["end"] + 1, tokens.size(1))
                        vec = extract_last_hidden(model, tokens[0, :end_pos], device)
                        store.write(vec.cpu(), surprise=chunk["mean_surprise"])
    store.age_all(tokens_processed=len(LEARN_FACTS))


def run_interfere_phase(model, store, tokenizer, device):
    """Phase 2: Feed unrelated noise text. Memory should remain robust."""
    model.eval()
    for noise_text in INTERFERE_NOISE:
        tokens = torch.tensor(tokenizer.encode(noise_text), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            model(tokens)
    store.age_all(tokens_processed=len(INTERFERE_NOISE))


def compute_recall_score(model, store, tokenizer, device, learn_hiddens):
    """Phase 3: Query each recall question and compare hidden-state similarity
    to the hidden states captured during the LEARN phase.

    Returns:
        mean cosine similarity between recall hidden states and their corresponding
        learn hidden states (higher = better recall).
    """
    model.eval()
    similarities = []
    for i, query_text in enumerate(RECALL_QUERIES):
        tokens = torch.tensor(tokenizer.encode(query_text), dtype=torch.long).to(device)
        recall_hidden = extract_last_hidden(model, tokens, device)
        # Compare to the learn-phase hidden for the corresponding fact
        if i < len(learn_hiddens):
            sim = F.cosine_similarity(
                recall_hidden.unsqueeze(0).float(),
                learn_hiddens[i].unsqueeze(0).float()
            ).item()
            similarities.append(sim)
    return sum(similarities) / len(similarities) if similarities else 0.0


def capture_learn_hiddens(model, tokenizer, device):
    """Capture hidden-state fingerprints for each fact during a clean forward pass."""
    model.eval()
    hiddens = []
    for fact_text in LEARN_FACTS:
        tokens = torch.tensor(tokenizer.encode(fact_text), dtype=torch.long).to(device)
        h = extract_last_hidden(model, tokens, device)
        hiddens.append(h.cpu())
    return hiddens


def run_single_config(name, cellmem_cfg, model, tokenizer, device):
    """Run the full 3-phase protocol for a single configuration.

    Returns a dict with config name, recall score, and memory stats.
    """
    d_model = model.config.n_embd

    # Rebuild model cellmem state for this config
    if cellmem_cfg is not None:
        # Create fresh store for this config
        store = MemoryStore(cellmem_cfg, d_model=d_model)
        calc = SurpriseCalculator(cellmem_cfg)
        model.memory_store = store
    else:
        store = None
        model.memory_store = None

    # Capture learn hidden states (reference: clean forward without memory interference)
    learn_hiddens = capture_learn_hiddens(model, tokenizer, device)

    # Phase 1: LEARN
    if store is not None:
        run_learn_phase(model, store, calc, tokenizer, device)
        memories_written = store.active_count
    else:
        memories_written = 0

    # Phase 2: INTERFERE
    if store is not None:
        run_interfere_phase(model, store, tokenizer, device)

    # Phase 3: RECALL
    recall_score = compute_recall_score(
        model, store, tokenizer, device, learn_hiddens
    )

    # Cleanup
    model.memory_store = None

    return {
        "name": name,
        "recall_score": recall_score,
        "memories_written": memories_written,
    }


def print_results_table(results):
    """Print a formatted comparison table."""
    baseline_score = None
    for r in results:
        if r["name"] == "baseline":
            baseline_score = r["recall_score"]
            break

    header = f"{'Config':<22} {'Recall':>8} {'Delta':>8} {'Mem Slots':>10}"
    print("\n" + "=" * len(header))
    print("CellMem v2 Eval Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        delta = r["recall_score"] - baseline_score if baseline_score is not None else 0.0
        delta_str = f"{delta:+.4f}" if r["name"] != "baseline" else "---"
        print(f"{r['name']:<22} {r['recall_score']:>8.4f} {delta_str:>8} {r['memories_written']:>10}")
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="CellMem v2 evaluation: 7-config comparison")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to nanochat checkpoint directory")
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Model tag within checkpoint dir (default: auto-detect largest)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    args = parser.parse_args()

    # Auto-detect device
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
    model, tokenizer, meta_data = load_model_from_dir(
        args.checkpoint, device, phase="eval", model_tag=args.model_tag
    )
    model.eval()
    print(f"Model loaded: n_layer={model.config.n_layer}, n_embd={model.config.n_embd}")

    # Run each configuration
    results = []
    for name, cellmem_cfg in CONFIGS.items():
        print(f"\nRunning config: {name}...")
        result = run_single_config(name, cellmem_cfg, model, tokenizer, device)
        results.append(result)
        print(f"  recall={result['recall_score']:.4f}, memories={result['memories_written']}")

    # Print comparison table
    print_results_table(results)


if __name__ == "__main__":
    main()
