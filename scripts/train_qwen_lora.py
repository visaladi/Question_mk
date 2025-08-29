import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# Stay offline
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "scripts" / "data" / "qwen_sft.jsonl"
BASE_MODEL_DIR = ROOT / "hf_models" / "qwen2.5-5b"        # your local base model
OUT_DIR = ROOT / "models" / "qwen2.5-5b-lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert BASE_MODEL_DIR.exists(), f"Missing base model: {BASE_MODEL_DIR}"
assert DATA_FILE.exists(), f"Missing dataset: {DATA_FILE}"

use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if use_bf16 else torch.float16

# 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=DTYPE,
)

# LoRA config for Qwen
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

def main():
    print("CUDA:", use_cuda, "| GPU:", torch.cuda.get_device_name(0) if use_cuda else "-")

    tok = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_DIR), local_files_only=True, use_fast=True, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    offload_dir = ROOT / "offload"
    offload_dir.mkdir(exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=bnb_config,   # <-- 4-bit
        device_map="auto",
        torch_dtype=DTYPE,
        offload_folder=str(offload_dir),  # safe CPU/NVMe spillover
        offload_state_dict=True,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(model, lora_config)

    # Convert chat messages -> plain text via chat template
    ds = load_dataset("json", data_files=str(DATA_FILE))["train"]
    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

    MAX_LEN = 1536 if use_cuda else 1024
    def tok_fn(batch):
        enc = tok(batch["text"], max_length=MAX_LEN, padding="max_length", truncation=True)
        enc["labels"] = enc["input_ids"].copy()
        return enc
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,     # sim batch=8
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=use_cuda and not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",          # CUDA-only optimizer
        report_to=[],
        dataloader_pin_memory=True,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    print(f"✅ LoRA saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
