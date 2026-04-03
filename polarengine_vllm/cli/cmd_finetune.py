"""polarquant finetune — QLoRA fine-tuning + re-quantize.

Usage:
    polarquant finetune google/gemma-4-31B-it --dataset HuggingFaceH4/ultrachat_200k
"""

from __future__ import annotations


def run_finetune(args):
    print(f"🧊 PolarQuant Fine-tune: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  LoRA rank: {args.lora_rank}")

    try:
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        print("\n  Install: pip install peft trl bitsandbytes")
        print("  Or use the Colab skill: /polarquant-finetune")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Load with QLoRA (BnB 4-bit for training)
    print("\nLoading model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Apply LoRA
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    from datasets import load_dataset
    print(f"\nLoading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split=f"train[:{args.max_samples}]")

    # Train
    training_args = SFTConfig(
        output_dir=args.output or "/tmp/polarquant_finetune",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        max_seq_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("\nTraining...")
    trainer.train()
    print("\n✅ Training complete!")

    if args.merge:
        print("\nMerging LoRA adapters...")
        model = model.merge_and_unload()
        print("  Merged. To re-quantize with PolarQuant:")
        print(f"  polarquant quantize {args.output or '/tmp/polarquant_finetune'} --upload")
