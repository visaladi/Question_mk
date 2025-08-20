import os, json, re
from openai import OpenAI
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
    elif backend == "ollama":
import requests
        self.requests = requests
        self.ollama_url = os.getenv("OLLAMA_URL","http://localhost:11434")
    else:
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(self.model)
    mdl = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype="auto", device_map="auto")
    self.pipe = pipeline("text-generation", model=mdl, tokenizer=tok)


def chat_json(self, system: str, user: str, temperature: float=0.2, max_tokens: int=1200):
    if self.backend == "openai":
        resp = self.client.chat.completions.create(
        model=self.model,
        temperature=temperature,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_tokens=max_tokens,
        )
    return json.loads(resp.choices[0].message.content)


    elif self.backend == "ollama":
        payload = {
        "model": self.model,
        "format": "json",
        "options": {"temperature": temperature},
        "messages": [
        {"role":"system","content":system},
        {"role":"user","content":user}
        ]
        }
    r = self.requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
return json.loads(data["message"]["content"])


else: # hf
prompt = f"""SYSTEM:\n{system}\n\nUSER:\n{user}\n\nReturn ONLY valid JSON."""
out = self.pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)[0]["generated_text"]
m = re.search(r'\{(?:[^{}]|(?R))*\}\s*$', out, flags=re.S)
if not m:
m = re.search(r'\{.*\}', out, flags=re.S)
if not m:
raise RuntimeError("Model did not return JSON. Try smaller count or different model.")
return json.loads(m.group(0))