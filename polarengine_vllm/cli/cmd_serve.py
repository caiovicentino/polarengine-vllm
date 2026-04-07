"""polarquant serve — OpenAI-compatible API with PQ5 weights + PolarQuant KV cache.

Usage:
    polarquant serve google/gemma-4-31B-it --port 8000
    polarquant serve caiovicentino1/model --kv-nbits 3 --port 8000
    curl http://localhost:8000/v1/chat/completions -d '{"messages":[...]}'
"""

from __future__ import annotations

import gc
import json
import math
import time
from threading import Thread
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# Transformers-compatible PolarQuant KV Cache
# ═══════════════════════════════════════════════════════════════════

def _build_polar_cache(num_layers, num_kv_heads, head_dim, nbits, device="cuda"):
    """Create a transformers-compatible PolarQuant KV cache.

    Returns a Cache subclass that works with model.generate(past_key_values=cache).
    Supports hybrid models (Qwen3.5 linear attention layers).
    """
    from polarengine_vllm.kv_cache import PolarKVConfig, PolarKVCache

    # Skip compression if head_dim is not power of 2
    can_quantize = head_dim > 0 and (head_dim & (head_dim - 1)) == 0

    if not can_quantize:
        print(f"  head_dim={head_dim} is not power of 2 — KV cache stays FP16")
        return None

    config = PolarKVConfig(
        nbits=nbits,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        residual_length=128,
    )
    polar_cache = PolarKVCache(config, device=device)

    # Build a transformers Cache wrapper
    try:
        from transformers.cache_utils import Cache
        cache_base = Cache
    except ImportError:
        cache_base = object

    # Hybrid support for Qwen3.5 linear attention layers
    class _HybridCacheLayer:
        def __init__(self):
            self.conv_states = None
            self.recurrent_states = None
            self.is_initialized = True

        def lazy_initialization(self, *args, **kwargs):
            pass

        def update_conv_state(self, conv_states, **kwargs):
            self.conv_states = conv_states
            return conv_states

        def update_recurrent_state(self, recurrent_states, **kwargs):
            self.recurrent_states = recurrent_states
            return recurrent_states

    class PolarServingCache(cache_base):
        """Transformers-compatible cache with PolarQuant KV compression."""

        def __init__(self, polar_kv):
            try:
                super().__init__()
            except (TypeError, ValueError):
                pass
            self._polar = polar_kv
            self._seen_tokens = 0
            # For hybrid models (Qwen3.5 linear attention)
            if not hasattr(self, "layers"):
                self.layers = [None] * num_layers

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[2]
            return self._polar.update(key_states, value_states, layer_idx)

        def get_seq_length(self, layer_idx=0):
            return self._polar.get_seq_length(layer_idx)

        def get_max_cache_shape(self):
            return None

        def get_mask_sizes(self, query_length, layer_idx):
            return query_length, self.get_seq_length(layer_idx)

        def has_previous_state(self, layer_idx=0):
            has_kv = self.get_seq_length(layer_idx) > 0
            has_linear = (
                self.layers[layer_idx] is not None
                and isinstance(self.layers[layer_idx], _HybridCacheLayer)
                and (self.layers[layer_idx].conv_states is not None
                     or self.layers[layer_idx].recurrent_states is not None)
            )
            return has_kv or has_linear

        # Hybrid model support (Qwen3.5 linear attention)
        def update_conv_state(self, conv_states, layer_idx, **kwargs):
            if self.layers[layer_idx] is None:
                self.layers[layer_idx] = _HybridCacheLayer()
            self.layers[layer_idx].conv_states = conv_states
            return conv_states

        def update_recurrent_state(self, recurrent_states, layer_idx, **kwargs):
            if self.layers[layer_idx] is None:
                self.layers[layer_idx] = _HybridCacheLayer()
            self.layers[layer_idx].recurrent_states = recurrent_states
            return recurrent_states

        @property
        def seen_tokens(self):
            return self._seen_tokens

        def __getitem__(self, idx):
            k_layer = self._polar.k_layers[idx]
            v_layer = self._polar.v_layers[idx]
            if k_layer is None:
                return (None, None)
            return (k_layer.residual, v_layer.residual)

        def __len__(self):
            return num_layers

        def __iter__(self):
            for i in range(num_layers):
                yield self[i]

        def reset(self):
            self._polar.reset()
            self._seen_tokens = 0
            self.layers = [None] * num_layers

        def memory_mb(self):
            return self._polar.memory_mb()

        def stats(self):
            return self._polar.stats()

    return PolarServingCache(polar_cache)


# ═══════════════════════════════════════════════════════════════════
# Main serve entry point
# ═══════════════════════════════════════════════════════════════════

def run_serve(args):
    """Start an OpenAI-compatible API server with PQ5 weights + KV cache compression."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("Install dependencies: pip install fastapi uvicorn")
        return

    from .cmd_chat import _load_model_streaming
    from transformers import TextIteratorStreamer

    # ── Load model ──────────────────────────────────────────────
    vision = getattr(args, "vision", False)
    model, tokenizer, processor, num_layers, num_kv_heads, head_dim = \
        _load_model_streaming(args.model, vision=vision)

    model_vram = torch.cuda.memory_allocated() / 1e9

    # ── Setup KV cache ──────────────────────────────────────────
    no_kv = getattr(args, "no_kv_cache", False)
    kv_nbits = None if no_kv else getattr(args, "nbits_kv", None)
    kv_cache_template = None
    kv_info = "FP16 (default)"

    if kv_nbits:
        print(f"\nInitializing PolarQuant Q{kv_nbits} KV cache...")
        print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")
        kv_cache_template = _build_polar_cache(
            num_layers, num_kv_heads, head_dim, kv_nbits
        )
        if kv_cache_template is not None:
            ratio = 16.0 / kv_nbits
            kv_info = f"PolarQuant Q{kv_nbits} ({ratio:.1f}x compression)"
            print(f"  KV cache: {kv_info}")
        else:
            kv_nbits = None

    # ── FastAPI app ─────────────────────────────────────────────
    app = FastAPI(title="PolarQuant API", version="0.6.0")

    def _make_cache():
        """Create a fresh KV cache for each request."""
        if kv_cache_template is None:
            return None
        kv_cache_template.reset()
        return kv_cache_template

    @app.get("/health")
    def health():
        vram = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        info = {
            "status": "ok",
            "model": args.model,
            "weights": "PQ5 + INT4",
            "kv_cache": kv_info,
            "vram_gb": round(vram, 1),
            "vram_peak_gb": round(peak, 1),
            "model_vram_gb": round(model_vram, 1),
        }
        return info

    @app.get("/v1/models")
    def list_models():
        return {"data": [{"id": args.model, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        stream = request.get("stream", False)

        # Tokenize
        chat_out = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        input_ids = chat_out["input_ids"] if hasattr(chat_out, "input_ids") else chat_out
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to("cuda")
        prompt_len = input_ids.shape[1]

        # Fresh KV cache per request
        cache = _make_cache()

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            repetition_penalty=1.3,
        )
        if cache is not None:
            gen_kwargs["past_key_values"] = cache
            gen_kwargs["use_cache"] = True

        if stream:
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer
            t_start = time.time()

            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            def event_stream():
                n_tokens = 0
                for text in streamer:
                    n_tokens += 1
                    chunk = {
                        "id": f"chatcmpl-polar",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {"content": text},
                            "index": 0,
                            "finish_reason": None,
                        }],
                        "model": args.model,
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk
                elapsed = time.time() - t_start
                tps = n_tokens / elapsed if elapsed > 0 else 0
                final = {
                    "id": f"chatcmpl-polar",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                    "model": args.model,
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": n_tokens,
                        "tokens_per_second": round(tps, 1),
                    },
                }
                if cache is not None:
                    final["kv_cache"] = {
                        "method": f"PolarQuant Q{kv_nbits}",
                        "memory_mb": round(cache.memory_mb(), 1),
                        "seq_length": cache.get_seq_length(),
                    }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        else:
            t_start = time.time()
            with torch.no_grad():
                out = model.generate(**gen_kwargs)
            elapsed = time.time() - t_start

            new_tokens = out[0][prompt_len:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            n_gen = len(new_tokens)
            tps = n_gen / elapsed if elapsed > 0 else 0

            result = {
                "id": f"chatcmpl-polar",
                "object": "chat.completion",
                "choices": [{
                    "message": {"role": "assistant", "content": response_text},
                    "index": 0,
                    "finish_reason": "stop",
                }],
                "model": args.model,
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": n_gen,
                    "tokens_per_second": round(tps, 1),
                },
            }
            if cache is not None:
                result["kv_cache"] = {
                    "method": f"PolarQuant Q{kv_nbits}",
                    "memory_mb": round(cache.memory_mb(), 1),
                    "seq_length": cache.get_seq_length(),
                }
            return result

    # ── Launch ──────────────────────────────────────────────────
    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 8000)

    print(f"\n{'='*60}")
    print(f"  PolarQuant API Server")
    print(f"{'='*60}")
    print(f"  Model:     {args.model}")
    print(f"  Weights:   PQ5 + INT4 (torchao)")
    print(f"  KV Cache:  {kv_info}")
    print(f"  VRAM:      {model_vram:.1f} GB")
    print(f"  Endpoint:  http://{host}:{port}")
    print(f"{'='*60}")
    print()
    print(f"  Test:")
    print(f"  curl http://localhost:{port}/v1/chat/completions \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"messages":[{{"role":"user","content":"Hello!"}}], "stream":true}}\'')
    print()

    uvicorn.run(app, host=host, port=port)
