#!/usr/bin/env python3
"""
CellMem v3 web chat server with architectural memory (HS injection).

Serves a chat UI with CellMem v3 memory backed by Qwen3-4B-Instruct.
Every message is automatically memorized via hidden states.
With --lora-ckpt, uses HS injection (memory inside architecture).
Without, falls back to text prefix injection.

Usage:
  python scripts/chat_web_v3.py --device cuda --port 7860 \
    --router-ckpt /tmp/cellmem_v3_ckpt/router_final.pt \
    --lora-ckpt /tmp/cellmem_v3_ckpt/lora_final.pt

SSH tunnel: ssh -L 7860:localhost:7860 <cluster>
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanochat.cellmem_v3 import CellMemConfig
from scripts.train_cellmem_qwen_v3 import CellMemWrapper

parser = argparse.ArgumentParser(description='CellMem v3 Web Chat')
parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--router-ckpt", default=None)
parser.add_argument("--lora-ckpt", default=None)
parser.add_argument("--lora-rank", type=int, default=16)
parser.add_argument("--memory-dir", default="/tmp/cellmem_v3_memory")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_args()

# Global state
wrapper: CellMemWrapper = None
memory_dir: Path = Path(args.memory_dir)
use_hs_injection: bool = False


def save_memory():
    if wrapper is None:
        return
    memory_dir.mkdir(parents=True, exist_ok=True)
    wrapper.store.save(memory_dir / "memory.pt")


def load_memory():
    mem_path = memory_dir / "memory.pt"
    if mem_path.exists() and wrapper is not None:
        wrapper.store.load(mem_path)
        n = wrapper.store.active_episodes
        print(f"  [memory loaded: {n} episodes]")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global wrapper, use_hs_injection
    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to(args.device)

    n_layers = model.config.num_hidden_layers
    layer_indices = list(range(n_layers - 4, n_layers))

    config = CellMemConfig(
        router_layers=layer_indices, episode_size=8, top_k=4, surprise_threshold=2.0,
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
        use_hs_injection = True
        print(f"  HS injection mode enabled (LoRA on {n_layers} layers)")

    load_memory()
    mode = "HS injection" if use_hs_injection else "text prefix"
    print(f"Server ready at http://localhost:{args.port} [{mode}]")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def root():
    ui_path = Path(__file__).parent.parent / "nanochat" / "ui_v3.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>CellMem v3</h1><p>ui_v3.html not found</p>")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


def _generate_hs_streaming(user_msg: str, max_tokens: int, temperature: float):
    """Generate with HS injection: prefill with hooks, then stream from cache."""
    prompt = wrapper._format_query_only(user_msg)
    inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)
    hooks = wrapper.inject_memory_hs(user_msg)

    if hooks:
        with torch.no_grad():
            prefill_out = wrapper.base_model(
                input_ids=inputs["input_ids"], use_cache=True)
        wrapper._remove_hooks(hooks)
        past_kv = prefill_out.past_key_values
        last_token = prefill_out.logits[:, -1:].argmax(dim=-1)
    else:
        with torch.no_grad():
            prefill_out = wrapper.base_model(
                input_ids=inputs["input_ids"], use_cache=True)
        past_kv = prefill_out.past_key_values
        last_token = prefill_out.logits[:, -1:].argmax(dim=-1)

    # First token from prefill
    first_id = last_token.item()
    if first_id == wrapper.tokenizer.eos_token_id:
        return
    generated_ids = [first_id]
    first_text = wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True)
    if first_text:
        yield first_text
    prev_text = first_text
    for _ in range(max_tokens - 1):
        with torch.no_grad():
            out = wrapper.base_model(
                input_ids=last_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[:, -1].clone()
        # Repetition penalty
        if generated_ids:
            for prev_id in set(generated_ids[-50:]):
                logits[0, prev_id] /= 1.3
        if temperature > 0.01:
            probs = torch.softmax(logits / temperature, dim=-1)
            last_token = torch.multinomial(probs, 1)
        else:
            last_token = logits.argmax(dim=-1, keepdim=True)
        if last_token.item() == wrapper.tokenizer.eos_token_id:
            break
        generated_ids.append(last_token.item())
        # Decode full sequence to avoid broken UTF-8 multi-byte chars
        full_text = wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_text = full_text[len(prev_text):]
        if new_text:
            yield new_text
        prev_text = full_text

    # Memorize after generation
    answer = wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if answer:
        wrapper.write_memory_full(f"User: {user_msg}\nAssistant: {answer}")
        save_memory()


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if not wrapper:
        raise HTTPException(503, "Model not loaded")

    user_msg = None
    for m in reversed(request.messages):
        if m.role == "user":
            user_msg = m.content
            break
    if not user_msg:
        raise HTTPException(400, "No user message")

    max_tokens = request.max_tokens or args.max_tokens
    temperature = max(request.temperature or 0.7, 0.01)

    if use_hs_injection:
        # HS injection: token-by-token from prefill cache
        gen = _generate_hs_streaming(user_msg, max_tokens, temperature)

        async def stream_hs():
            for text in gen:
                yield f"data: {json.dumps({'token': text}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(stream_hs(), media_type="text/event-stream")
    else:
        # Fallback: text prefix + TextIteratorStreamer
        from transformers import TextIteratorStreamer
        prompt = wrapper.retrieve_and_format(user_msg)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)
        streamer = TextIteratorStreamer(
            wrapper.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            pad_token_id=wrapper.tokenizer.eos_token_id, streamer=streamer)
        thread = Thread(target=wrapper.base_model.generate, kwargs=gen_kwargs)
        thread.start()

        async def stream_text():
            full_response = []
            for text in streamer:
                if text:
                    full_response.append(text)
                    yield f"data: {json.dumps({'token': text}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'done': True})}\n\n"
            answer = "".join(full_response).strip()
            if answer:
                wrapper.write_memory_full(f"User: {user_msg}\nAssistant: {answer}")
                save_memory()

        return StreamingResponse(stream_text(), media_type="text/event-stream")


@app.get("/memory")
async def memory_status():
    if not wrapper:
        return {"status": "not loaded"}
    s = wrapper.store
    texts = [t for t in s.episode_texts[:s.active_episodes] if t]
    slots = []
    for i, t in enumerate(texts):
        slots.append({"slot": i, "text": t[:250], "tokens": []})
    return {
        "cellmem": "enabled",
        "mode": "hs_injection" if use_hs_injection else "text_prefix",
        "active_slots": s.active_count,
        "total_slots": s.config.n_slots,
        "active_episodes": s.active_episodes,
        "slots": slots,
        "edges": [],
    }


@app.post("/memory/reset")
async def memory_reset():
    if wrapper:
        wrapper.clear_memory()
        mem_path = memory_dir / "memory.pt"
        if mem_path.exists():
            mem_path.unlink()
    return {"status": "memory reset", "slots": 0}


@app.get("/health")
async def health():
    return {"status": "ok", "ready": wrapper is not None,
            "mode": "hs_injection" if use_hs_injection else "text_prefix"}


if __name__ == "__main__":
    import uvicorn
    mode = "HS injection" if args.lora_ckpt else "text prefix"
    print(f"Starting CellMem v3 Web Server [{mode}] on port {args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
