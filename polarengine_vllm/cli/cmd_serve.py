"""polarquant serve — OpenAI-compatible API server.

Usage:
    polarquant serve google/gemma-4-31B-it --port 8000
    curl http://localhost:8000/v1/chat/completions -d '{"messages":[...]}'
"""

from __future__ import annotations


def run_serve(args):
    """Start an OpenAI-compatible API server."""
    import torch
    import json
    import time
    from threading import Thread

    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        import uvicorn
    except ImportError:
        print("Install FastAPI: pip install fastapi uvicorn")
        return

    from .cmd_chat import _load_model_streaming
    from transformers import TextIteratorStreamer

    model, tokenizer, processor, num_layers, num_kv_heads, head_dim = \
        _load_model_streaming(args.model, vision=args.vision)

    app = FastAPI(title="PolarQuant API")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": args.model,
                "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1)}

    @app.get("/v1/models")
    def models():
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
        input_ids = input_ids.to("cuda")

        if stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                repetition_penalty=1.3,
                streamer=streamer,
            )
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            def event_stream():
                for text in streamer:
                    chunk = {
                        "choices": [{"delta": {"content": text}, "index": 0}],
                        "model": args.model,
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    repetition_penalty=1.3,
                )
            response_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            return {
                "choices": [{"message": {"role": "assistant", "content": response_text}, "index": 0}],
                "model": args.model,
                "usage": {"prompt_tokens": input_ids.shape[1],
                          "completion_tokens": out.shape[1] - input_ids.shape[1]},
            }

    print(f"\n🚀 PolarQuant API running on http://{args.host}:{args.port}")
    print(f"   Model: {args.model}")
    print(f"   VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"\n   Test: curl http://localhost:{args.port}/v1/chat/completions \\")
    print(f'     -H "Content-Type: application/json" \\')
    print(f'     -d \'{{"messages":[{{"role":"user","content":"Hello!"}}]}}\'')

    uvicorn.run(app, host=args.host, port=args.port)
