from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# >>> EDIT THIS to your absolute model folder (forward slashes on Windows)
MODEL_DIR = Path(r"C:/Users/visal Adikari/OneDrive/Desktop/uni sem 7/fyp/project/Question_mk/hf_models/qwen2.5-7b")
OFFLOAD_DIR = Path(r"C:/mlcache/offload"); OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("torch cuda version:", getattr(torch.version, "cuda", "none"))

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

if torch.cuda.is_available():
    # GPU path: 4-bit QLoRA loading with safe offload
    use_bf16 = torch.cuda.is_bf16_supported()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    try:
        m = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            offload_folder=str(OFFLOAD_DIR),
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        print("✅ Loaded 4-bit on GPU")
    except ValueError as e:
        # If VRAM is tight, constrain memory and try again
        print("First GPU load failed:", e)
        max_mem = {"cuda:0": "10GiB", "cpu": "32GiB"}  # adjust to your VRAM/RAM
        m = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=bnb,
            device_map="auto",
            max_memory=max_mem,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            offload_folder=str(OFFLOAD_DIR),
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        print("✅ Loaded 4-bit on GPU with max_memory/offload")
else:
    # CPU fallback: no quantization, no offload
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
    )
    print("✅ Loaded on CPU (no CUDA detected)")
