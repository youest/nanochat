"""
CellMem v2 evaluation script: multi-round A/B comparison.

Round 1 (default): baseline + 2 write strategies x 3 layer placements = 7 configs
Round 2: raw vs delta write mode for the most promising config = 3 configs

Protocol:
  Phase 1 (LEARN): Feed novel invented facts -> memories accumulate
  Phase 2 (INTERFERE): Feed unrelated noise text -> test memory robustness
  Phase 3 (RECALL): Query facts from Phase 1, measure recall via hidden-state cosine similarity

Usage:
    python -m scripts.eval_cellmem_v2 --checkpoint path/to/checkpoint_dir              # round 1
    python -m scripts.eval_cellmem_v2 --checkpoint path/to/checkpoint_dir --round=2    # raw vs delta
    python -m scripts.eval_cellmem_v2 --checkpoint path/to/checkpoint_dir --round=all  # everything

Importable:
    from scripts.eval_cellmem_v2 import CONFIGS_ROUND1, CONFIGS_ROUND2
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.cellmem_v2 import CellMemConfig, MemoryStore, SurpriseCalculator


# ---------------------------------------------------------------------------
# Evaluation configurations — organized in rounds to avoid combinatorial explosion.
#
# Round 1 (CORE): baseline + 2 write_strategies x 3 layer placements (all raw)
# Round 2 (DELTA): raw vs delta for the most promising config (per_token_last3)
#
# Run with --round=1 (default) or --round=2, or --round=all.
# ---------------------------------------------------------------------------
CONFIGS_ROUND1 = {
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

CONFIGS_ROUND2 = {
    "baseline": None,
    "raw_last3":   CellMemConfig(enabled=True, write_strategy="per_token", layers="last3",
                                  surprise_threshold=2.0, n_slots=64, write_mode="raw"),
    "delta_last3": CellMemConfig(enabled=True, write_strategy="per_token", layers="last3",
                                  surprise_threshold=2.0, n_slots=64, write_mode="delta"),
}

# Round 3 (DECORRELATION): test min_novelty (DG pattern separation) at different thresholds
# Uses P90 surprise threshold from previous eval (12.0) instead of 2.0
CONFIGS_ROUND3 = {
    "baseline": None,
    "no_decorr":   CellMemConfig(enabled=True, write_strategy="per_token", layers="mid",
                                  surprise_threshold=12.0, n_slots=64, min_novelty=0.0),
    "decorr_01":   CellMemConfig(enabled=True, write_strategy="per_token", layers="mid",
                                  surprise_threshold=12.0, n_slots=64, min_novelty=0.1),
    "decorr_02":   CellMemConfig(enabled=True, write_strategy="per_token", layers="mid",
                                  surprise_threshold=12.0, n_slots=64, min_novelty=0.2),
    "decorr_03":   CellMemConfig(enabled=True, write_strategy="per_token", layers="mid",
                                  surprise_threshold=12.0, n_slots=64, min_novelty=0.3),
}

# Legacy alias for imports
CONFIGS = CONFIGS_ROUND1

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


# ---------------------------------------------------------------------------
# Diagnostic measurements
# ---------------------------------------------------------------------------

def measure_surprise_distribution(model, tokenizer, device, texts=None):
    """Measure per-token surprise distribution across a set of texts.
    Returns all surprise values for histogram analysis."""
    if texts is None:
        texts = LEARN_FACTS + INTERFERE_NOISE
    model.eval()
    all_surprises = []
    for text in texts:
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tokens)
        if tokens.size(1) > 1:
            pred_logits = logits[:, :-1, :]
            targets = tokens[:, 1:]
            surprises = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            all_surprises.extend(surprises.cpu().tolist())
    return all_surprises


def measure_memory_quality(store):
    """Compute memory vector quality metrics.
    Returns dict with pairwise similarity stats and PCA dimensionality."""
    if store is None or store.active_count < 2:
        return {"active": 0, "mean_sim": 0, "min_sim": 0, "max_sim": 0,
                "pca_dim_80": 0, "pca_dim_95": 0}

    active = store.vectors[:store.active_count].float()

    # Pairwise cosine similarity
    norms = active.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = active / norms
    sim_matrix = normed @ normed.T
    # Extract upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
    pairwise_sims = sim_matrix[mask]

    # PCA: how many components for 80% and 95% variance?
    centered = active - active.mean(dim=0, keepdim=True)
    U, S, V = torch.svd(centered)
    variance_explained = (S ** 2) / (S ** 2).sum()
    cumvar = variance_explained.cumsum(0)
    pca_80 = (cumvar < 0.80).sum().item() + 1
    pca_95 = (cumvar < 0.95).sum().item() + 1

    return {
        "active": store.active_count,
        "mean_sim": pairwise_sims.mean().item(),
        "min_sim": pairwise_sims.min().item(),
        "max_sim": pairwise_sims.max().item(),
        "std_sim": pairwise_sims.std().item(),
        "pca_dim_80": pca_80,
        "pca_dim_95": pca_95,
    }


def measure_generative_recall(model, tokenizer, device, max_tokens=64):
    """Generate answers to recall queries and check for expected keywords.
    Returns per-query results and aggregate score."""
    model.eval()
    results = []
    for i, query in enumerate(RECALL_QUERIES):
        tokens = torch.tensor(tokenizer.encode(query), dtype=torch.long).unsqueeze(0).to(device)
        generated_tokens = []
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = model(tokens)
                next_logit = logits[:, -1, :]
                next_token = next_logit.argmax(dim=-1, keepdim=True)
                token_id = next_token.item()
                # Stop on special tokens
                decoded = tokenizer.decode([token_id])
                if '<|' in decoded:
                    break
                generated_tokens.append(token_id)
                tokens = torch.cat([tokens, next_token], dim=1)

        answer = tokenizer.decode(generated_tokens)
        expected = RECALL_KEYWORDS[i] if i < len(RECALL_KEYWORDS) else []
        hits = [kw for kw in expected if kw.lower() in answer.lower()]
        score = len(hits) / len(expected) if expected else 0.0
        results.append({
            "query": query,
            "answer": answer[:120],
            "hits": hits,
            "expected": expected,
            "score": score,
        })
    agg = sum(r["score"] for r in results) / len(results) if results else 0.0
    return results, agg


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
                recall_hidden.unsqueeze(0).cpu().float(),
                learn_hiddens[i].unsqueeze(0).cpu().float()
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


def _ensure_mem_gates(model, cellmem_cfg, device, force_gate=None):
    """Ensure model has mem_gates and _cellmem_layers for the given config.
    Base models trained without cellmem need these injected at eval time."""
    import torch.nn as nn
    from nanochat.gpt import _cellmem_layer_indices

    # Temporarily patch config to compute layer indices
    orig_cellmem = model.config.cellmem
    model.config.cellmem = cellmem_cfg
    layers = _cellmem_layer_indices(model.config)
    model._cellmem_layers = layers

    if layers:
        model.mem_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1, device=device)) for _ in layers
        ])
        if force_gate is not None:
            with torch.no_grad():
                for gate in model.mem_gates:
                    gate.fill_(force_gate)
    else:
        model.mem_gates = None

    model.config.cellmem = orig_cellmem


def run_single_config(name, cellmem_cfg, model, tokenizer, device, force_gate=None):
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
        # Ensure model has gates (needed for base models without cellmem)
        _ensure_mem_gates(model, cellmem_cfg, device, force_gate=force_gate)
    else:
        store = None
        model.memory_store = None
        model.mem_gates = None
        model._cellmem_layers = []

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

    # Memory quality metrics
    mem_quality = measure_memory_quality(store)

    # Generative recall (does the model actually produce correct answers?)
    gen_recall_results, gen_recall_score = measure_generative_recall(
        model, tokenizer, device
    )

    # Cleanup
    model.memory_store = None

    return {
        "name": name,
        "recall_score": recall_score,
        "gen_recall_score": gen_recall_score,
        "gen_recall_results": gen_recall_results,
        "memories_written": memories_written,
        "mem_quality": mem_quality,
    }


def print_results_table(results):
    """Print a formatted comparison table."""
    baseline_score = None
    baseline_gen = None
    for r in results:
        if r["name"] == "baseline":
            baseline_score = r["recall_score"]
            baseline_gen = r["gen_recall_score"]
            break

    header = f"{'Config':<20} {'HidRecall':>9} {'GenRecall':>9} {'Slots':>6} {'MeanSim':>8} {'PCA80':>6} {'PCA95':>6}"
    print("\n" + "=" * len(header))
    print("CellMem v2 Eval Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        mq = r["mem_quality"]
        mean_sim = f"{mq['mean_sim']:.3f}" if mq['active'] > 1 else "---"
        pca80 = f"{mq['pca_dim_80']}" if mq['active'] > 1 else "---"
        pca95 = f"{mq['pca_dim_95']}" if mq['active'] > 1 else "---"
        print(f"{r['name']:<20} {r['recall_score']:>9.4f} {r['gen_recall_score']:>9.4f} "
              f"{r['memories_written']:>6} {mean_sim:>8} {pca80:>6} {pca95:>6}")
    print("=" * len(header))

    # Print generative recall details for non-baseline configs
    for r in results:
        if r["name"] == "baseline" or not r.get("gen_recall_results"):
            continue
        print(f"\n--- Generative Recall: {r['name']} ---")
        for gr in r["gen_recall_results"]:
            status = "HIT" if gr["score"] > 0 else "MISS"
            print(f"  [{status}] Q: {gr['query'][:60]}")
            print(f"        A: {gr['answer'][:80]}")
            if gr["hits"]:
                print(f"        Keywords found: {gr['hits']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="CellMem v2 evaluation: multi-round comparison")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to nanochat checkpoint directory")
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Model tag within checkpoint dir (default: auto-detect largest)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--force-gate", type=float, default=None,
                        help="Force mem_gates to this value (bypasses sigmoid(-10) init)")
    parser.add_argument("--round", type=str, default="1", choices=["1", "2", "3", "all"],
                        help="Which eval round: 1=core, 2=delta, 3=decorrelation, all=everything")
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

    # Select configs based on round
    if args.round == "1":
        configs = CONFIGS_ROUND1
    elif args.round == "2":
        configs = CONFIGS_ROUND2
    elif args.round == "3":
        configs = CONFIGS_ROUND3
    elif args.round == "all":
        configs = {**CONFIGS_ROUND1,
                   **{k: v for k, v in CONFIGS_ROUND2.items() if k != "baseline"},
                   **{k: v for k, v in CONFIGS_ROUND3.items() if k != "baseline"}}

    print(f"Round {args.round}: {len(configs)} configurations")

    # Step 0: Measure surprise distribution to calibrate threshold
    print("\n--- Surprise Distribution Analysis ---")
    surprises = measure_surprise_distribution(model, tokenizer, device)
    surprises_sorted = sorted(surprises)
    n = len(surprises_sorted)
    print(f"  Total tokens: {n}")
    print(f"  Min: {surprises_sorted[0]:.2f}, Max: {surprises_sorted[-1]:.2f}")
    print(f"  Mean: {sum(surprises_sorted)/n:.2f}, Median: {surprises_sorted[n//2]:.2f}")
    for pct in [50, 75, 90, 95, 99]:
        idx = min(int(n * pct / 100), n - 1)
        print(f"  P{pct}: {surprises_sorted[idx]:.2f}")
    print(f"  Suggested threshold (P90): {surprises_sorted[min(int(n * 0.9), n-1)]:.2f}")
    print()

    # Run each configuration
    results = []
    for name, cellmem_cfg in configs.items():
        print(f"\nRunning config: {name}...")
        result = run_single_config(name, cellmem_cfg, model, tokenizer, device,
                                   force_gate=args.force_gate)
        results.append(result)
        print(f"  recall={result['recall_score']:.4f}, memories={result['memories_written']}")

    # Print comparison table
    print_results_table(results)


if __name__ == "__main__":
    main()
