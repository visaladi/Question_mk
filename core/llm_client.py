# core/llm_client.py
import os, json, re
from typing import Literal, Optional

import requests

from config.config import config

BAD_MODEL_VALUES = {None, "", " ", "string", "null", "none", "None", "String"}

def _sanitize_model(backend: str, model: Optional[str]) -> str:
    if model in BAD_MODEL_VALUES:
        if backend == "gemini":
            return "gemini-2.0-flash"
        elif backend == "ollama":
            # Default to your requested Ollama model
            return "qwen2.5:1.5b-instruct"
        else:
            # hf (HuggingFace local)
            return "Qwen/Qwen2.5-7B-Instruct"
    return model  # type: ignore

class LLMClient:
    """
    Backends:
      - gemini: Google Generative Language API (needs GEMINI_API_KEY)
      - ollama: local (http://localhost:11434)
      - hf: HuggingFace transformers (local)
    """
    def __init__(
        self,
        backend: Literal["gemini","ollama","hf"] = config.LLM_BACKEND,
        model: Optional[str] = config.LLM_MODEL
    ):
        self.backend = backend
        self.model = _sanitize_model(backend, model)
        self.requests = requests

        if backend == "gemini":
            self.api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set. Set it in env or config.py")
            self.base = "https://generativelanguage.googleapis.com/v1beta"

        elif backend == "ollama":
            # Allow override via env var; default to local daemon
            self.ollama_url = config.OLLAMA_URL or os.getenv("OLLAMA_URL", "http://localhost:11434")

        else:  # hf
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
            )
            self.pipe = pipeline("text-generation", model=mdl, tokenizer=tok)

    def _gemini_call(self, system: str, user: str, temperature: float, max_tokens: int):
        # Use response as JSON directly
        url = f"{self.base}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        # Ask Gemini to return JSON only
        payload = {
            "systemInstruction": {"role": "system", "parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "response_mime_type": "application/json",
                "responseMimeType": "application/json",
                "maxOutputTokens": max_tokens,
            },
        }
        r = self.requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        data = r.json()
        # Extract text from the first candidate
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise RuntimeError(f"Gemini returned unexpected response: {data}")
        # Ensure valid JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # last-resort extraction of a JSON object
            m = re.search(r'\{(?:[^{}]|(?R))*\}\s*$', text, flags=re.S)
            if not m:
                m = re.search(r'\{.*\}', text, flags=re.S)
            if not m:
                raise RuntimeError(f"Gemini did not return JSON. Raw: {text[:400]}")
            return json.loads(m.group(0))

    def chat_json(self, system: str, user: str, temperature: float=0.2, max_tokens: int=1200):
        if self.backend == "gemini":
            return self._gemini_call(system, user, temperature, max_tokens)


        elif self.backend == "ollama":

            payload = {

                "model": self.model,

                "format": "json",  # ask model to return JSON content

                "stream": False,  # <<< IMPORTANT: single JSON object instead of NDJSON stream

                "options": {"temperature": temperature},

                "messages": [

                    {"role": "system", "content": system},

                    {"role": "user", "content": user},

                ],

            }

            try:

                r = self.requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=600)

                r.raise_for_status()

            except requests.exceptions.ConnectionError as e:

                raise RuntimeError(

                    f"Ollama not reachable at {self.ollama_url}. Run 'ollama serve' and ensure the model is pulled."

                ) from e

            # Primary path: single JSON object

            try:

                data = r.json()

                content = data.get("message", {}).get("content", "")

            except ValueError:

                # Fallback: handle NDJSON (if some Ollama versions ignore stream=False)

                texts = []

                for line in r.text.splitlines():

                    line = line.strip()

                    if not line:
                        continue

                    try:

                        obj = json.loads(line)

                        msg = obj.get("message", {}).get("content")

                        if msg:
                            texts.append(msg)

                    except Exception:

                        continue

                content = "".join(texts)

            if not content:
                snippet = r.text[:300].replace("\n", "\\n")

                raise RuntimeError(f"Ollama returned no content. Raw: {snippet}")

            # Ensure the model's content is valid JSON (strip code fences if present)

            try:

                return json.loads(content)

            except json.JSONDecodeError:

                # ```json ... ``` fences

                m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content, flags=re.S | re.I)

                if m:
                    return json.loads(m.group(1))

                # largest { ... } block heuristic

                m = re.search(r'\{[\s\S]*\}', content, flags=re.S)

                if m:
                    return json.loads(m.group(0))

                raise RuntimeError(f"Model did not return JSON. Raw: {content[:400]}")


        else:  # hf
            out = self.pipe(
                f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nReturn ONLY valid JSON.",
                max_new_tokens=max_tokens, do_sample=True, temperature=temperature
            )[0]["generated_text"]
            m = re.search(r'\{(?:[^{}]|(?R))*\}\s*$', out, flags=re.S)
            if not m:
                m = re.search(r'\{.*\}', out, flags=re.S)
            if not m:
                raise RuntimeError("Model did not return JSON. Try a different model or reduce count.")
            return json.loads(m.group(0))
