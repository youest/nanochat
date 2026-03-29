#!/usr/bin/env python3
"""
CellMem v3 interactive chat with architectural memory (HS injection).

The model automatically memorizes everything from the conversation.
Hidden states are captured for ALL tokens and injected into every
layer's attention — memory lives inside the architecture, not as text.

Memory persists across sessions via auto-save/load.

Commands:
  /clear          Clear all memories (also deletes saved file)
  /stats          Show memory stats
  /quit           Exit (memory auto-saved)

Usage:
  python scripts/chat_cellmem_v3.py --device cuda
  python scripts/chat_cellmem_v3.py --router-ckpt /tmp/cellmem_v3_ckpt/router_final.pt \
                                     --lora-ckpt /tmp/cellmem_v3_ckpt/lora_final.pt
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

DEFAULT_MEMORY_DIR = Path("/tmp/cellmem_v3_memory")


def save_memory(wrapper: CellMemWrapper, memory_dir: Path):
    memory_dir.mkdir(parents=True, exist_ok=True)
    wrapper.store.save(memory_dir / "memory.pt")
    n = wrapper.store.active_episodes
    print(f"  [memory saved: {n} episodes]")


def load_memory(wrapper: CellMemWrapper, memory_dir: Path) -> bool:
    mem_path = memory_dir / "memory.pt"
    if mem_path.exists():
        wrapper.store.load(mem_path)
        n = wrapper.store.active_episodes
        texts = [t for t in wrapper.store.episode_texts[:n] if t]
        print(f"  [memory loaded: {n} episodes, {len(texts)} texts]")
        return True
    return False


def generate_with_hs(wrapper: CellMemWrapper, user_input: str,
                     max_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate response using hidden state injection (prefill + cache)."""
    # Format prompt
    prompt = wrapper._format_query_only(user_input)
    inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)

    # Inject memory hidden states via hooks
    hooks = wrapper.inject_memory_hs(user_input)

    if hooks:
        # Prefill with HS injection → KV cache captures memory
        with torch.no_grad():
            prefill_out = wrapper.base_model(
                input_ids=inputs["input_ids"],
                use_cache=True,
            )
        wrapper._remove_hooks(hooks)

        # Generate from enriched cache
        past_kv = prefill_out.past_key_values
        last_token = prefill_out.logits[:, -1:].argmax(dim=-1)
        generated_ids = [last_token]
        for _ in range(max_tokens - 1):
            with torch.no_grad():
                out = wrapper.base_model(
                    input_ids=last_token,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            past_kv = out.past_key_values
            if temperature > 0:
                probs = torch.softmax(out.logits[:, -1] / temperature, dim=-1)
                last_token = torch.multinomial(probs, 1)
            else:
                last_token = out.logits[:, -1:].argmax(dim=-1)
            generated_ids.append(last_token)
            if last_token.item() == wrapper.tokenizer.eos_token_id:
                break

        gen_tokens = torch.cat(generated_ids, dim=1)
        return wrapper.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
    else:
        # No memory — generate normally
        with torch.no_grad():
            gen_ids = wrapper.base_model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 0.01),
                top_p=0.9, pad_token_id=wrapper.tokenizer.eos_token_id,
            )
        answer_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
        return wrapper.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="CellMem v3 chat with architectural memory")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--router-ckpt", default=None, help="Path to trained router checkpoint")
    parser.add_argument("--lora-ckpt", default=None, help="Path to trained LoRA checkpoint")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--memory-dir", type=str, default=str(DEFAULT_MEMORY_DIR),
                        help="Directory for persistent memory storage")
    args = parser.parse_args()
    memory_dir = Path(args.memory_dir)

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
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

    if args.lora_ckpt:
        print(f"Installing LoRA (rank={args.lora_rank}) and loading weights...")
        wrapper.install_memory_lora(rank=args.lora_rank, all_layers=True)
        lora_state = torch.load(args.lora_ckpt, map_location=args.device, weights_only=True)
        for layer_idx in range(n_layers):
            attn = model.model.layers[layer_idx].self_attn
            if hasattr(attn, '_original_q_proj'):
                key_a = f"layer_{layer_idx}_lora_A"
                key_b = f"layer_{layer_idx}_lora_B"
                if key_a in lora_state:
                    attn.q_proj.lora_A.load_state_dict(lora_state[key_a])
                    attn.q_proj.lora_B.load_state_dict(lora_state[key_b])
        print(f"  LoRA loaded for {n_layers} layers")

    # Load persisted memory
    load_memory(wrapper, memory_dir)

    mode = "HS injection" if args.lora_ckpt else "text prefix"
    print(f"\nCellMem v3 Chat — {mode}")
    print(f"Everything you say is memorized. Memory persists across sessions.")
    print(f"Memory dir: {memory_dir}")
    print(f"Commands: /clear, /stats, /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            save_memory(wrapper, memory_dir)
            print("Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            save_memory(wrapper, memory_dir)
            print("Bye!")
            break

        if user_input.lower() == "/clear":
            wrapper.clear_memory()
            mem_path = memory_dir / "memory.pt"
            if mem_path.exists():
                mem_path.unlink()
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

        # Generate with HS injection (architectural memory)
        if args.lora_ckpt:
            answer = generate_with_hs(wrapper, user_input, max_tokens=args.max_tokens)
        else:
            answer = wrapper.generate_with_memory(
                user_input, max_new_tokens=args.max_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        print(f"Bot: {answer}\n")

        # Memorize full turn (all tokens for HS injection)
        wrapper.write_memory_full(f"User: {user_input}\nAssistant: {answer}")

        # Auto-save after each turn
        save_memory(wrapper, memory_dir)


if __name__ == "__main__":
    main()
