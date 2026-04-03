"""polarquant quantize — quantize a model with PolarQuant Q5+INT4.

Usage:
    polarquant quantize google/gemma-4-31B-it --upload
    polarquant quantize Qwen/Qwen3.5-9B --output caiovicentino1/model
"""

from __future__ import annotations


def run_quantize(args):
    """Quantize a model and optionally upload to HuggingFace."""
    # Reuse the streaming loader from cmd_chat
    from .cmd_chat import _load_model_streaming
    import torch
    import os
    import json

    model, tokenizer, processor, num_layers, num_kv_heads, head_dim = \
        _load_model_streaming(args.model, vision=args.vision)

    # Quick sanity test
    print("\nSanity test...")
    if processor:
        msgs = [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                                return_tensors="pt", add_generation_prompt=True).to("cuda")
    else:
        msgs = [{"role": "user", "content": "What is 2+2?"}]
        chat_out = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
        input_ids = chat_out["input_ids"] if hasattr(chat_out, "input_ids") else chat_out
        inputs = {"input_ids": input_ids.to("cuda")}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Model says: {response}")

    # Save
    save_dir = args.save_dir or "/tmp/polarquant_output"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving to {save_dir}...")
    save_path = os.path.join(save_dir, "model_int4.pt")
    torch.save(model.state_dict(), save_path)
    sz = os.path.getsize(save_path) / 1e9
    print(f"  Model: {sz:.1f} GB")

    # Config + tokenizer
    model.config.save_pretrained(save_dir)
    if processor:
        processor.save_pretrained(save_dir)
    else:
        tokenizer.save_pretrained(save_dir)

    # polar_config.json
    vram = torch.cuda.memory_allocated() / 1e9
    polar_config = {
        "quantization_method": "PolarQuant",
        "version": "v5",
        "weight_format": "torchao_int4",
        "text_weight_bits": 4,
        "vision_weight_bits": 16 if args.vision else None,
        "kv_cache_bits": 3,
        "block_size": 128,
        "head_dim": head_dim,
        "base_model": args.model,
        "multimodal": args.vision,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "vram_gb": round(vram, 1),
    }
    with open(os.path.join(save_dir, "polar_config.json"), "w") as f:
        json.dump(polar_config, f, indent=2)

    print(f"\n  Saved: {save_dir}")
    print(f"  VRAM: {vram:.1f} GB")

    # Upload
    if args.upload:
        from huggingface_hub import HfApi

        repo = args.output
        if not repo:
            short = args.model.split("/")[-1]
            suffix = "-PolarQuant-Q5-Vision" if args.vision else "-PolarQuant-Q5"
            repo = f"caiovicentino1/{short}{suffix}"

        print(f"\nUploading to {repo}...")
        api = HfApi()
        api.create_repo(repo, exist_ok=True)
        api.upload_file(
            path_or_fileobj=save_path,
            path_in_repo="model_int4.pt",
            repo_id=repo,
            repo_type="model",
        )
        api.upload_folder(
            folder_path=save_dir,
            repo_id=repo,
            repo_type="model",
            allow_patterns=["*.json", "*.jinja", "tokenizer*", "*.model"],
        )

        pipeline = "image-text-to-text" if args.vision else "text-generation"
        print(f"\n✅ https://huggingface.co/{repo}")
        print(f"   Pipeline: {pipeline}")
    else:
        print(f"\n  To upload: polarquant quantize {args.model} --upload")
