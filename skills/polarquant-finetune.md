# PolarQuant Fine-tune Pipeline

Generate a Colab notebook (.ipynb) that fine-tunes a PolarQuant model with QLoRA/LoRA, then re-quantizes and uploads. Full pipeline: load PQ5 → dequant → LoRA train → merge → re-quantize PQ5+INT4 → upload.

## Input

The user provides a model name and dataset. Examples:
- `google/gemma-4-31B-it` with `HuggingFaceH4/ultrachat_200k`
- `caiovicentino1/Qwen3.5-9B-PolarQuant-Q5` with custom dataset

## Instructions

1. **Parse the model**: Extract specs via WebFetch.
2. **Determine LoRA config**: Based on model size (rank, alpha, target modules).
3. **Generate notebook**: Write `.ipynb` to `~/Desktop/POLARQUANT_FINETUNE_{SHORT_NAME}.ipynb`.

## Notebook Structure (9 cells)

### Cell 0: Markdown header

### Cell 1: Install
```python
!pip install git+https://github.com/huggingface/transformers.git --force-reinstall -q
!pip install -q accelerate safetensors sentencepiece scipy torchao
!pip install -q peft trl datasets bitsandbytes
```

### Cell 2: Load base model in BF16

Load the base model (NOT quantized) for LoRA training:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# For training, use BnB QLoRA (4-bit base + LoRA adapters)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_config,
    dtype=torch.bfloat16, attn_implementation='sdpa',
)
```

### Cell 3: Apply LoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

# LoRA config scaled by model size
lora_config = LoraConfig(
    r=16,              # rank (16 for ≤9B, 32 for 27B+)
    lora_alpha=32,     # alpha = 2*r
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Cell 4: Prepare dataset

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset(DATASET, split='train[:10000]')

# Format as chat messages using tokenizer's chat template
def format_chat(example):
    messages = example['messages']  # adapt based on dataset format
    return {'text': tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_chat)
```

### Cell 5: Train

```python
training_args = SFTConfig(
    output_dir='/content/lora_output',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    max_seq_length=2048,
    dataset_text_field='text',
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Cell 6: Merge LoRA + PolarQuant re-quantize

```python
# Merge LoRA adapters into base weights
model = model.merge_and_unload()

# Now re-quantize with PolarQuant Q5 + torchao INT4
# Use the streaming approach from /polarquant-inference
# ... PQ5 dequant + INT4 per-module ...
```

**CRITICAL**: After merging LoRA, the model is in BF16 (BnB dequanted + LoRA merged). Apply PolarQuant Q5 quantize+dequant, then torchao INT4.

### Cell 7: Test fine-tuned model

Quick generation test to verify fine-tuning worked.

### Cell 8: Save + Upload

```python
# Save INT4 model
torch.save(model.state_dict(), '/content/model_finetuned_int4.pt')

# Upload to HF
REPO = f'caiovicentino1/{Model-Name}-finetuned-PolarQuant-Q5'
api.create_repo(REPO, exist_ok=True)
api.upload_file(...)
```

## Key Patterns

- **BnB QLoRA for training** (4-bit base, BF16 LoRA adapters) — most memory efficient
- **PolarQuant for inference** (after merge) — best quality per bit
- **LoRA rank**: 16 for ≤9B, 32 for 27B+, 64 for 70B+
- **Target all linear layers** for best quality
- **SFTTrainer from TRL** — simplest fine-tuning API
- **Merge before re-quantize** — LoRA must be merged into base weights
- **Test before upload** — verify fine-tuning didn't break the model
- **BF16 throughout** — never .half()

## Output

Tell the user:
1. File path of the notebook
2. LoRA config (rank, target modules)
3. Estimated training time
4. VRAM requirement for training vs inference
5. Upload repo name

## Argument: $ARGUMENTS
