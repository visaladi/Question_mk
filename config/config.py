# config.py
from dataclasses import dataclass

@dataclass
class AppConfig:
    # Static backend/model selection
    LLM_BACKEND: str = "gemini"   # "gemini" | "ollama" | "hf"
    LLM_MODEL: str   = "gemini-2.0-flash"

    # Directly put your Gemini API key here
    GEMINI_API_KEY: str = "AIzaSyAzIrDUnqNO35ZrWTnwJEb5TEXleNpX08w"

    # Reranker toggle (optional)
    USE_RERANK: bool = False

    # Server bind
    HOST: str = "127.0.0.2"
    PORT: int = 8000

config = AppConfig()
