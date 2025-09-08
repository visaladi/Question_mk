# config.py
from dataclasses import dataclass
import os

@dataclass
class AppConfig:
    # Static backend/model selection
    LLM_BACKEND: str = "gemini"   # "gemini" | "ollama" | "hf" | "lora"
    LLM_MODEL = os.getenv("MODEL_NAME", "qwen2.5:1.5b-instruct")  # "gemini-2.0-flash" | "Qwen/Qwen2.5-7B-Instruct" | "qwen2.5:1.5b-instruct"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # Reranker toggle (optional)
    USE_RERANK: bool = False
    # Path to a LoRA adapter (local dir or HF repo id), e.g. "your-user/qwen2.5-mcq-lora"
    LORA_ADAPTER: str = "" # "" disables LoRA
    # Merge LoRA weights into the base model and free memory (slightly faster inference)
    LORA_MERGE_AND_UNLOAD: bool = False
    # Load the base model in 4-bit (requires bitsandbytes). Great for small GPUs.
    LORA_4BIT: bool = False

    # Server bind
    HOST: str = "127.0.0.2"
    PORT: int = 8000

config = AppConfig()
