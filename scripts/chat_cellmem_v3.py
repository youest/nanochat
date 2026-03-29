#!/usr/bin/env python3
"""
CellMem v3 interactive chat with automatic memory.

The model automatically memorizes everything from the conversation.
Each user message and bot response is written to memory via the
surprise-gated write path. When answering, the router retrieves
relevant memories and injects them as context.

Commands:
  /clear          Clear all memories
  /stats          Show memory stats
  /quit           Exit

Usage:
  python scripts/chat_cellmem_v3.py --device cuda
  python scripts/chat_cellmem_v3.py --router-ckpt /tmp/cellmem_v3_ckpt/router_final.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanochat.cellmem_v3 import CellMemConfig
from scripts.train_cellmem_qwen_v3 import CellMemWrapper


def main():
    parser = argparse.ArgumentParser(description="CellMem v3 chat with automatic memory")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--router-ckpt", default=None, help="Path to trained router checkpoint")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to(args.device)

    n_layers = model.config.num_hidden_layers
    layer_indices = list(range(n_layers - 4, n_layers))

    config = CellMemConfig(
        router_layers=layer_indices,
        episode_size=8,
        top_k=4,
        surprise_threshold=2.0,
    )
    wrapper = CellMemWrapper(model, tokenizer, layer_indices, config=config, device=args.device)

    if args.router_ckpt:
        print(f"Loading router from {args.router_ckpt}")
        state = torch.load(args.router_ckpt, map_location=args.device, weights_only=True)
        wrapper.router.load_state_dict(state)

    print(f"\nCellMem v3 Chat — automatic memory")
    print(f"Everything you say is memorized. The model retrieves relevant memories automatically.")
    print(f"Commands: /clear, /stats, /quit\n")

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Bye!")
            break

        if user_input.lower() == "/clear":
            wrapper.clear_memory()
            turn = 0
            print("Memory cleared.\n")
            continue

        if user_input.lower() == "/stats":
            s = wrapper.store
            texts = [t for t in s.episode_texts[:s.active_episodes] if t]
            print(f"  Episodes: {s.active_episodes}")
            print(f"  Tokens stored: {s.active_count}")
            print(f"  Memories: {len(texts)}")
            for i, t in enumerate(texts):
                print(f"    [{i}] {t[:80]}{'...' if len(t) > 80 else ''}")
            print()
            continue

        # 1. Memorize user message automatically
        wrapper.write_memory(f"User said: {user_input}")
        turn += 1

        # 2. Retrieve relevant memories and generate response
        prompt = wrapper.retrieve_and_format(user_input)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        print(f"Bot: {answer}\n")

        # 3. Memorize bot response too
        wrapper.write_memory(f"Assistant said: {answer}")


if __name__ == "__main__":
    main()
