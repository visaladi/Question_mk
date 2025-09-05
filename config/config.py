# config.py
from dataclasses import dataclass

@dataclass
class AppConfig:
    # Static backend/model selection
    LLM_BACKEND: str = "gemini"   # "gemini" | "ollama" | "hf" | "lora"
    #LLM_MODEL: str   = "qwen2.5:1.5b-instruct" # "gemini-2.0-flash" | "Qwen/Qwen2.5-7B-Instruct" | "qwen2.5:1.5b-instruct"
    LLM_MODEL = "gemini-2.0-flash"
    # Directly put your Gemini API key here
    GEMINI_API_KEY: str = "AIzaSyAzIrDUnqNO35ZrWTnwJEb5TEXleNpX08w"

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
