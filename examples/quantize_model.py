"""Quantize a HuggingFace model to PolarQuant format.

Usage:
    python quantize_model.py --model Qwen/Qwen3.5-9B --output ./polar-9b/
"""

import argparse

from polarengine_vllm.quantizer import PolarQuantizer


def main():
    parser = argparse.ArgumentParser(description="Quantize a model with PolarEngine")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./Qwen3.5-9B-PolarEngine/",
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for quantization (default: 128)",
    )
    parser.add_argument(
        "--no-pack-int4",
        action="store_true",
        help="Disable INT4 nibble packing for Q3/Q4 layers",
    )
    args = parser.parse_args()

    quantizer = PolarQuantizer(block_size=args.block_size)
    quantizer.quantize_model(
        model_name=args.model,
        output_dir=args.output,
        pack_int4=not args.no_pack_int4,
    )
    print(f"Done! Serve with: vllm serve {args.output} --quantization polarengine")


if __name__ == "__main__":
    main()
