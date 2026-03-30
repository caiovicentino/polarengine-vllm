"""Serve a PolarEngine quantized model with vLLM.

Usage:
    python serve_model.py --model ./Qwen3.5-9B-PolarEngine/
"""

import argparse

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Serve a PolarEngine model with vLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="./Qwen3.5-9B-PolarEngine/",
        help="Path to quantized model directory",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    args = parser.parse_args()

    model = LLM(
        model=args.model,
        quantization="polarengine",
        dtype="half",
        max_model_len=args.max_model_len,
    )

    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to check if a number is prime:",
    ]

    params = SamplingParams(temperature=0.7, max_tokens=200)
    outputs = model.generate(prompts, params)

    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")


if __name__ == "__main__":
    main()
