#!/usr/bin/env python3
"""
CellMem v3 web chat server with automatic memory.

Serves a chat UI with CellMem v3 memory backed by Qwen3-4B-Instruct.
Every message is automatically memorized. The router retrieves relevant
memories and injects them as context for each response.

Memory persists across server restarts via auto-save.

Usage:
  python scripts/chat_web_v3.py --device cuda --port 7860
  python scripts/chat_web_v3.py --router-ckpt /tmp/cellmem_v3_ckpt/router_final.pt

Then open http://localhost:7860 in browser.
SSH tunnel: ssh -L 7860:localhost:7860 <cluster>
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
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
parser.add_argument("--memory-dir", default="/tmp/cellmem_v3_memory")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_args()

# Global state
wrapper: CellMemWrapper = None
memory_dir: Path = Path(args.memory_dir)


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
    global wrapper
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

    load_memory()
    print(f"Server ready at http://localhost:{args.port}")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve chat UI from file."""
    ui_path = Path(__file__).parent.parent / "nanochat" / "ui_v3.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    # Fallback: minimal UI
    return HTMLResponse(content="<h1>CellMem v3</h1><p>ui_v3.html not found</p>")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if not wrapper:
        raise HTTPException(503, "Model not loaded")

    # Get the latest user message
    user_msg = None
    for m in reversed(request.messages):
        if m.role == "user":
            user_msg = m.content
            break
    if not user_msg:
        raise HTTPException(400, "No user message")

    # Hybrid MSA: text prefix + KV cache injection
    prompt = wrapper.retrieve_and_format(user_msg)
    inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.device)

    # KV cache injection
    mem_cache = wrapper.inject_memory_kv(user_msg)

    # Stream generation
    streamer = TextIteratorStreamer(wrapper.tokenizer, skip_prompt=True, skip_special_tokens=True)
    max_tokens = request.max_tokens or args.max_tokens

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=max(request.temperature or 0.7, 0.01),
        top_p=0.9,
        pad_token_id=wrapper.tokenizer.eos_token_id,
        streamer=streamer,
    )

    if mem_cache is not None:
        n_mem = mem_cache.get_seq_length()
        seq_len = inputs["input_ids"].shape[1]
        gen_kwargs["past_key_values"] = mem_cache
        gen_kwargs["attention_mask"] = torch.ones(
            1, n_mem + seq_len, device=wrapper.device, dtype=torch.long)
        gen_kwargs["position_ids"] = torch.arange(
            n_mem, n_mem + seq_len, device=wrapper.device).unsqueeze(0)
        gen_kwargs["cache_position"] = torch.arange(
            n_mem, n_mem + seq_len, device=wrapper.device)
    else:
        gen_kwargs["attention_mask"] = inputs.get("attention_mask")

    thread = Thread(target=wrapper.base_model.generate, kwargs=gen_kwargs)
    thread.start()

    async def stream_response():
        full_response = []
        for text in streamer:
            if text:
                full_response.append(text)
                yield f"data: {json.dumps({'token': text}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

        yield f"data: {json.dumps({'done': True})}\n\n"

        # Memorize full turn (user + bot together for context)
        answer = "".join(full_response).strip()
        if answer:
            wrapper.write_memory(f"User: {user_msg}\nAssistant: {answer}")
            save_memory()

    return StreamingResponse(stream_response(), media_type="text/event-stream")


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
    return {"status": "ok", "ready": wrapper is not None}


if __name__ == "__main__":
    import uvicorn
    print(f"Starting CellMem v3 Web Server on port {args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
