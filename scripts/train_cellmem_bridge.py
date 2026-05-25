#!/usr/bin/env python3
"""MemoryBridge experiment — latent memory prefix on a frozen backbone.

Tests the thesis (docs/cellmem-v4/2026-05-25-memory-bridge-experiment.md):
a trained, lightweight bridge that injects memory as a soft prefix at the
*input embedding* level lets a FROZEN backbone answer from memory — matching
the text-prefix baseline on single-fact recall. This is the result none of the
7 prior architectural attempts achieved (KV injection = 0%, HS injection =
loops).

Backbone frozen; only MemoryBridge (gist tokens + W1/W2/W3) is trained.
Architecture-agnostic: uses only get_input_embeddings / inputs_embeds /
output_hidden_states, so it applies to Qwen3.5's hybrid Gated-DeltaNet stack.

Usage:
  # 5-min load verification on the GPU box (no training):
  python scripts/train_cellmem_bridge.py --smoke --model Qwen/Qwen3.5-4B

  # train the bridge + eval head-to-head vs text_only:
  python scripts/train_cellmem_bridge.py --model Qwen/Qwen3.5-4B \
      --device cuda --k-gist 16 --steps 400 --out /tmp/bridge_ckpt
"""
from __future__ import annotations
import argparse
import contextlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanochat.memory_bridge import MemoryBridge, answer_logit_span
from scripts.train_cellmem_qwen_v3 import (
    FICTIONAL_FACTS, CONVERSATIONAL_MEMORY,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset() -> list[dict]:
    """Same items used by the v3 baseline: {memory, query, answer}."""
    data = []
    for memory, query, answer in FICTIONAL_FACTS:
        data.append({"memory": memory, "query": query, "answer": answer})
    for memory, query, answer in CONVERSATIONAL_MEMORY:
        data.append({"memory": memory, "query": query, "answer": answer})
    return data


# ---------------------------------------------------------------------------
# Trainer / wrapper
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def maybe_autocast(device: str):
    if device != "cpu" and torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


class BridgeTrainer:
    def __init__(self, model, tokenizer, k_gist: int, device: str):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.embed = model.get_input_embeddings()
        self.model_dtype = next(model.parameters()).dtype
        d = model.config.hidden_size
        self.bridge = MemoryBridge(d_model=d, k_gist=k_gist).to(device)  # fp32
        self.k_gist = k_gist
        # Freeze the backbone — only the bridge trains.
        for p in model.parameters():
            p.requires_grad_(False)

    # -- prompt formatting (mirrors the v3 text_only baseline) --------------

    def _apply_template(self, messages, add_generation_prompt=True) -> str:
        """apply_chat_template, disabling Qwen thinking mode when supported."""
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return self.tok.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def _query_prompt(self, query: str) -> str:
        q_text = f"Question: {query}\nAnswer:"
        return self._apply_template([{"role": "user", "content": q_text}])

    def _text_prompt(self, query: str, memory: str) -> str:
        system = ("You have access to memories. Use them to answer accurately.\n"
                  f"Memory 1: {memory}")
        q_text = f"Question: {query}\nAnswer:"
        return self._apply_template([
            {"role": "system", "content": system},
            {"role": "user", "content": q_text},
        ])

    def _ids(self, text: str, add_special: bool) -> torch.Tensor:
        return self.tok(text, return_tensors="pt",
                        add_special_tokens=add_special).input_ids.to(self.device)

    # -- core mechanism ------------------------------------------------------

    def encode_memory(self, memory: str) -> torch.Tensor:
        """memory string -> latent prefix [1, K, d] (bridge-projected)."""
        ids = self._ids(memory, add_special=True)            # [1, M]
        mem_emb = self.embed(ids)                            # [1, M, d]
        gist = self.bridge.gist_embeds(1).to(mem_emb.dtype)  # [1, K, d]
        full = torch.cat([mem_emb, gist], dim=1)             # [1, M+K, d]
        out = self.model(inputs_embeds=full, output_hidden_states=True,
                         use_cache=False)
        h_gist = out.hidden_states[-1][:, -self.k_gist:, :]  # [1, K, d]
        return self.bridge.project(h_gist.float())           # fp32 [1, K, d]

    def bridge_loss(self, item: dict) -> torch.Tensor:
        """CE on the answer tokens, conditioned on [latent prefix ; query]."""
        prefix = self.encode_memory(item["memory"])          # fp32 [1,K,d]
        prefix = prefix.to(self.model_dtype)

        prompt_ids = self._ids(self._query_prompt(item["query"]), add_special=False)
        answer_ids = self._ids(" " + item["answer"], add_special=False)
        q_len, a_len = prompt_ids.shape[1], answer_ids.shape[1]

        seq_emb = torch.cat([
            prefix,
            self.embed(prompt_ids),
            self.embed(answer_ids),
        ], dim=1)                                            # [1, K+Q+A, d]
        out = self.model(inputs_embeds=seq_emb, use_cache=False)
        start, end = answer_logit_span(self.k_gist, q_len, a_len)
        pred = out.logits[0, start:end, :].float()           # [A, V]
        return F.cross_entropy(pred, answer_ids[0])

    # -- generation ----------------------------------------------------------

    @torch.no_grad()
    def gen_text_only(self, query: str, memory: str, max_new=30) -> str:
        ids = self._ids(self._text_prompt(query, memory), add_special=False)
        gen = self.model.generate(input_ids=ids, max_new_tokens=max_new,
                                  do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def gen_embed_prefix(self, query: str, memory: str, max_new=30) -> str:
        prefix = self.encode_memory(memory).to(self.model_dtype)  # [1,K,d]
        prompt_ids = self._ids(self._query_prompt(query), add_special=False)
        seq_emb = torch.cat([prefix, self.embed(prompt_ids)], dim=1)
        attn = torch.ones(seq_emb.shape[:2], dtype=torch.long, device=self.device)
        gen = self.model.generate(inputs_embeds=seq_emb, attention_mask=attn,
                                  max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        # With inputs_embeds (decoder-only) generate returns only new tokens.
        return self.tok.decode(gen[0], skip_special_tokens=True).strip()

    # -- eval ----------------------------------------------------------------

    @torch.no_grad()
    def score_recall(self, data: list[dict], mode: str, n: int = 20,
                     debug: bool = False) -> float:
        correct = 0
        for item in data[:n]:
            if mode == "text_only":
                gen = self.gen_text_only(item["query"], item["memory"])
            elif mode == "embed_prefix":
                gen = self.gen_embed_prefix(item["query"], item["memory"])
            else:
                raise ValueError(mode)
            hit = item["answer"].lower() in gen.lower()
            correct += int(hit)
            if debug:
                print(f"  {'OK' if hit else '..'} [{mode}] q={item['query'][:38]!r} "
                      f"exp={item['answer']!r} got={gen[:48]!r}")
        return correct / max(n, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(tr: BridgeTrainer, data: list[dict], steps: int, lr: float,
          device: str, accum: int = 8) -> None:
    """Train the bridge with gradient accumulation.

    Run 1 used batch=1 and showed catastrophic interference (answers borrowed
    from other memories). Accumulating grads over `accum` items before each
    optimizer step averages those competing gradients, which is the fix for
    batch-1 forgetting. We backward per item (freeing each graph) so we never
    hold `accum` copies of the 4B model's activation graph at once.
    """
    import random
    opt = torch.optim.AdamW(tr.bridge.parameters(), lr=lr)
    tr.bridge.train()
    for step in range(steps):
        opt.zero_grad()
        running = 0.0
        for _ in range(accum):
            item = random.choice(data)
            with maybe_autocast(device):
                loss = tr.bridge_loss(item)
            (loss / accum).backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(tr.bridge.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {running / accum:.4f}")
    tr.bridge.train(False)


# ---------------------------------------------------------------------------
# Smoke: 5-min load verification on the box (no training)
# ---------------------------------------------------------------------------

def run_smoke(model, tok, device: str) -> None:
    print("=== SMOKE: load verification ===")
    d = model.config.hidden_size
    emb = model.get_input_embeddings()
    print(f"hidden_size={d}  embedding.weight={tuple(emb.weight.shape)}")
    assert emb.weight.shape[1] == d

    ids = tok("Hello world", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    print(f"forward OK  logits={tuple(out.logits.shape)}  "
          f"n_hidden_layers={len(out.hidden_states)}")
    assert out.hidden_states[-1].shape[-1] == d

    # inputs_embeds path (what the bridge uses)
    with torch.no_grad():
        out2 = model(inputs_embeds=emb(ids), output_hidden_states=True, use_cache=False)
    print(f"inputs_embeds forward OK  logits={tuple(out2.logits.shape)}")

    # chat template + thinking mode
    msgs = [{"role": "user", "content": "Question: What is 2+2?\nAnswer:"}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        print("apply_chat_template(enable_thinking=False) OK")
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print("apply_chat_template OK (no enable_thinking kwarg)")
    print(f"prompt repr (head): {p[:160]!r}")

    # generate with inputs_embeds (embed_prefix path)
    pe = emb(tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device))
    attn = torch.ones(pe.shape[:2], dtype=torch.long, device=device)
    with torch.no_grad():
        g = model.generate(inputs_embeds=pe, attention_mask=attn,
                           max_new_tokens=12, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    print(f"generate(inputs_embeds) returned shape={tuple(g.shape)} "
          f"text={tok.decode(g[0], skip_special_tokens=True)!r}")
    print("=== SMOKE OK ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k-gist", type=int, default=16)
    ap.add_argument("--steps", type=int, default=150,
                    help="optimizer updates (each averages --accum items)")
    ap.add_argument("--accum", type=int, default=8,
                    help="gradient accumulation: items averaged per update")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument("--out", default="/tmp/bridge_ckpt")
    ap.add_argument("--smoke", action="store_true",
                    help="load verification only, then exit")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code)
    model = model.to(args.device)
    model.train(False)

    if args.smoke:
        run_smoke(model, tok, args.device)
        return

    data = build_dataset()
    tr = BridgeTrainer(model, tok, k_gist=args.k_gist, device=args.device)
    n_params = sum(p.numel() for p in tr.bridge.parameters())
    print(f"MemoryBridge trainable params: {n_params:,} (backbone frozen)")

    # Apples-to-apples check: the two arms must differ only by how memory is
    # injected (system text vs latent prefix), not by thinking-mode scaffolding.
    print("\n[prompt audit for data[0]]")
    print(f"  text_only  prompt: {tr._text_prompt(data[0]['query'], data[0]['memory'])!r}")
    print(f"  embed arm  prompt: {tr._query_prompt(data[0]['query'])!r}")

    print("\n[baseline before training]")
    base_text = tr.score_recall(data, "text_only", n=args.eval_n)
    base_embed = tr.score_recall(data, "embed_prefix", n=args.eval_n)
    print(f"  text_only    = {base_text:.2%}")
    print(f"  embed_prefix = {base_embed:.2%} (untrained bridge)")

    print(f"\n[training bridge: {args.steps} updates x accum {args.accum} "
          f"= {args.steps * args.accum} item-forwards, lr={args.lr}]")
    train(tr, data, steps=args.steps, lr=args.lr, device=args.device, accum=args.accum)

    print("\n[eval after training]")
    text = tr.score_recall(data, "text_only", n=args.eval_n, debug=True)
    embed = tr.score_recall(data, "embed_prefix", n=args.eval_n, debug=True)
    print(f"\n=== RESULT (K={args.k_gist}) ===")
    print(f"  text_only    = {text:.2%}  (baseline)")
    print(f"  embed_prefix = {embed:.2%}  (thesis: should match text_only)")
    verdict = "PASS" if embed >= 0.75 else "FAIL"
    print(f"  thesis (embed_prefix >= 75%): {verdict}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(tr.bridge.state_dict(), out / "bridge.pt")
    print(f"  saved bridge -> {out / 'bridge.pt'}")


if __name__ == "__main__":
    main()
