import json, os
from pathlib import Path

# Paths
# Paths
ROOT = Path(__file__).resolve().parents[1]
MERGED = Path(r"C:\Users\visal Adikari\OneDrive\Desktop\uni sem 7\fyp\project\Question_mk\scripts\data\merged_output.json")
OUT_DIR = Path(r"C:\Users\visal Adikari\OneDrive\Desktop\uni sem 7\fyp\project\Question_mk\scripts\data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "qwen_sft.jsonl"

# Simple system prompts (kept strict to JSON)
MCQ_SYSTEM = "You are a precise question generator. Only output valid JSON for MCQs."
ESSAY_SYSTEM = "You are a precise question generator. Only output valid JSON for essay questions."

def infer_type(items):
    """MCQ if first item has 'options' and 'answer_index', else essay."""
    first = items[0]
    return "mcq" if isinstance(first, dict) and "options" in first and "answer_index" in first else "essay"

def build_user_prompt(qtype: str, n: int, difficulty: str | None):
    if qtype == "mcq":
        return (
            f"TASK: Create {n} MCQs at {difficulty or 'medium'} difficulty from the given CONTEXT.\n"
            "Return JSON with shape: {\"items\":[{\"question\":str,\"options\":[str,str,str,str],"
            "\"answer_index\":int,\"rationale\":str,\"bloom\":str,\"difficulty\":str,"
            "\"source_pages\":[int,...]}]}.\n"
            "CONTEXT: [omitted during training]"
        )
    return (
        f"TASK: Create {n} essay questions from the given CONTEXT with rubric bullets and target keywords.\n"
        "Return JSON with shape: {\"items\":[{\"question\":str,\"rubric_bullets\":[str,...],"
        "\"target_keywords\":[str,...],\"bloom\":str,\"difficulty\":str,\"source_pages\":[int,...]}]}.\n"
        "CONTEXT: [omitted during training]"
    )

def as_chat(system: str, user: str, assistant_obj: dict):
    # We store assistant as a JSON string so the model learns to emit JSON
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)}
        ]
    }

def main():
    if not MERGED.exists():
        raise FileNotFoundError(f"Missing {MERGED}. Place merged_output.json at repo root.")
    data = json.loads(MERGED.read_text(encoding="utf-8"))

    wrote = 0
    with OUT.open("w", encoding="utf-8") as wf:
        for batch in data:
            if not isinstance(batch, dict):
                continue
            items = batch.get("items") or []
            if not items:
                continue

            qtype = infer_type(items)
            difficulty = items[0].get("difficulty") if isinstance(items[0].get("difficulty"), str) else "medium"
            n = len(items)
            system = MCQ_SYSTEM if qtype == "mcq" else ESSAY_SYSTEM
            user = build_user_prompt(qtype, n, difficulty)

            # Normalize to {"items":[...]} exactly as ground-truth
            assistant = {"items": items}
            wf.write(json.dumps(as_chat(system, user, assistant), ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Wrote {wrote} training examples to {OUT}")

if __name__ == "__main__":
    main()
