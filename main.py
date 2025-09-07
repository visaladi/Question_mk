import os, json, tempfile, datetime, uvicorn, time, re, mlflow
from typing import Literal
from fastapi.responses import FileResponse, Response
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from config.config import config
from core.pdf_utils import extract_pages, chunk_pages
from core.embed_store import EmbedStore
from core.schemas import MCQBatch, EssayBatch
from core.llm_client import LLMClient
from core.prompts import MCQ_SYSTEM, ESSAY_SYSTEM
from pathlib import Path
from core.tracking import start_run, log_params, log_metrics, log_text, log_json, log_artifact


try:
    from extras.rerank import Reranker
except Exception:
    Reranker = None

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "front_end" / "static"
TEMPLATE_DIR = BASE_DIR / "front_end" / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="PDF → Questions (One-Shot)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

_ALLOWED_BLOOM = {"Remember","Understand","Apply","Analyze","Evaluate","Create"}
_BLOOM_SYNONYMS = {
    # common alternates -> canonical
    "comprehension": "Understand",
    "understanding": "Understand",
    "knowledge": "Remember",
    "remembering": "Remember",
    "application": "Apply",
    "analysis": "Analyze",
    "synthesis": "Create",
    "creation": "Create",
    "evaluating": "Evaluate",
    "evaluation": "Evaluate",
}
def _norm_bloom(v):
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {x.lower() for x in _ALLOWED_BLOOM}:
            return t.capitalize()
        if t in _BLOOM_SYNONYMS:
            return _BLOOM_SYNONYMS[t]
    return "Understand"  # safe default

def _norm_difficulty(v):
    # Map a bunch of synonyms → {"easy","medium","hard"}
    EASY = {"easy","beginner","basic","foundation","intro","introductory","elementary"}
    MED  = {"medium","med","moderate","intermediate","avg","average","normal","standard"}
    HARD = {"hard","difficult","advanced","challenging","tough","complex"}

    if isinstance(v, str):
        t = v.strip().lower()
        if t in EASY:  return "easy"
        if t in MED:   return "medium"
        if t in HARD:  return "hard"
    elif isinstance(v, (int, float)):
        # Optional numeric mapping
        if v <= 1: return "easy"
        if v >= 3: return "hard"
        return "medium"
    return "medium"


def _norm_answer_index(item):
    """
    Coerce answer_index into 0..3. Supports:
      - int (0..3 or 1..4)
      - str: 'A'/'B'/'C'/'D', '1'..'4', 'Option 2', the exact correct option text in 'answer'/'correct'/'solution'
    """
    idx = item.get("answer_index", None)
    opts = item.get("options", [])
    # If correct answer is provided as text, map to its index
    for key in ("answer", "correct", "correct_option", "solution"):
        if idx is None and isinstance(item.get(key), str) and opts:
            try:
                pos = [o.strip() for o in opts].index(item[key].strip())
                return pos
            except ValueError:
                pass

    # Strings like 'B', '2', 'Option 3'
    if isinstance(idx, str):
        s = idx.strip().upper()
        m = re.search(r'([A-D])', s)
        if m:
            return "ABCD".index(m.group(1))
        m = re.search(r'(\d+)', s)
        if m:
            n = int(m.group(1))
            if n in (1,2,3,4):
                return n-1
        # last resort: try exact text match
        if opts and s in [o.strip().upper() for o in opts]:
            return [o.strip().upper() for o in opts].index(s)

    # Ints 1..4 or 0..3
    if isinstance(idx, int):
        if 0 <= idx <= 3:
            return idx
        if 1 <= idx <= 4:
            return idx - 1

    return None  # let downstream validator catch if still invalid

def _sanitize_option_texts(options):
    out = []
    for o in options[:4]:  # keep first 4 only
        if isinstance(o, str):
            # remove leading "A) ", "1. ", etc.
            out.append(re.sub(r'^\s*([A-D]|\d+)[\.\)]\s*', '', o).strip())
        else:
            out.append(str(o))
    return out
# --- end helpers ---

@app.get("/health")
def health():
    return {
        "ok": True,
        "backend": config.LLM_BACKEND,
        "model": config.LLM_MODEL,
        "rerank_enabled": config.USE_RERANK and Reranker is not None,
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = STATIC_DIR / "favicon.ico"
    if path.exists():
        return FileResponse(str(path))
    # fallback if file missing (prevents 404 spam)
    return Response(status_code=204)

@app.post("/generate_once")
async def generate_once(
    pdf: UploadFile = File(...),
    qtype: Literal["mcq", "essay"] = Form(...),
    difficulty: Literal["easy", "medium", "hard"] = Form("medium"),
    count: int = Form(5),
    topic: str = Form("full document"),
):
    # 1) Validate + save uploaded file
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a .pdf file.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(await pdf.read())
        path = tf.name

        try:
            run_name = f"{qtype}_{difficulty}_{count}"
            with start_run(run_name, tags={
                "endpoint": "/generate_once",
                "model_backend": config.LLM_BACKEND,
                "model_name": config.LLM_MODEL,
            }):
                t_extract = time.time()
                pages = extract_pages(path)
                if not pages:
                    raise HTTPException(400, "No text found in the PDF.")
                chunks = list(chunk_pages(pages, max_chars=1800, overlap=200))  
                if not chunks:
                    raise HTTPException(400, "Could not create chunks from the PDF content.")
                
                log_params({
                    "qtype": qtype,
                    "difficulty": difficulty,
                    "count": count,
                    "topic": topic,
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "backend": config.LLM_BACKEND,
                    "model": config.LLM_MODEL,
                })
                log_metrics({"t_extract_sec": time.time() - t_extract})

                # build embed store + retrieve
                t_ret = time.time()
                store = EmbedStore()
                store.build(chunks)
                ctx_snips = store.topk(topic, k=6)
                if config.USE_RERANK and Reranker is not None:
                    reranker = Reranker()
                    ctx_snips = reranker.rerank(topic, ctx_snips, top_k=4)
                else:
                    ctx_snips = ctx_snips[:4]

                log_metrics({"t_retrieve_sec": time.time() - t_ret})
                log_text("\n\n".join([s[0] for s in ctx_snips]), "artifacts/context_snippets.txt")

                # build context
                context = "\n\n".join([f"[pages {','.join(map(str,p))}]\n{t}" for t, p in ctx_snips])

                # LLM call
                t_llm = time.time()
                llm = LLMClient(backend=config.LLM_BACKEND, model=config.LLM_MODEL)

                # 7) Generate
                if qtype == "mcq":
                    user_template = mlflow.genai.load_prompt("prompts:/quiz-generator-mcq/1")
                    llm_temperature = 0.2
                    llm_max_tokens = 1600  # a bit higher helps Qwen produce multiple items

                    want = int(count)
                    acc: list[dict] = []
                    seen_q = set()
                    max_rounds = 3
                    round_no = 0

                    while len(acc) < want and round_no < max_rounds:
                        need = want - len(acc)
                        user = user_template.format(context=context, n=need, difficulty=difficulty)

                        # call LLM API
                        payload = llm.chat_json(MCQ_SYSTEM, user, temperature=llm_temperature, max_tokens=llm_max_tokens)

                        # --- normalize model output BEFORE Pydantic ---
                        items = payload.get("items", []) if isinstance(payload, dict) else []
                        norm_items = []
                        for it in items:
                            it = dict(it)

                            # normalize bloom & difficulty
                            it["bloom"] = _norm_bloom(it.get("bloom"))
                            it["difficulty"] = _norm_difficulty(it.get("difficulty"))

                            # clean options & coerce answer_index
                            if "options" in it and isinstance(it["options"], list):
                                it["options"] = _sanitize_option_texts(it["options"])

                            ai = _norm_answer_index(it)
                            if ai is not None:
                                it["answer_index"] = ai

                            # basic shape checks
                            q = (it.get("question") or "").strip()
                            opts = it.get("options") or []
                            ai = it.get("answer_index", None)

                            if not q or len(opts) < 2:
                                continue

                            # ensure exactly 4 options by padding with plausible distractors
                            if len(opts) < 4:
                                base = set(o.strip() for o in opts if isinstance(o, str))
                                while len(opts) < 4:
                                    candidate = f"None of the other choices {len(opts)}"
                                    if candidate not in base:
                                        opts.append(candidate)
                                        base.add(candidate)
                                it["options"] = opts[:4]

                            # keep only valid 0..3 index
                            if isinstance(ai, int) and 0 <= ai < 4:
                                qkey = q.lower()
                                if qkey not in seen_q:
                                    seen_q.add(qkey)
                                    norm_items.append(it)

                        acc.extend(norm_items)
                        round_no += 1

                    # if still short, trim or raise
                    if not acc:
                        raise HTTPException(500, "Model returned no valid MCQs; try a smaller context or easier difficulty.")

                    payload = {"items": acc[:want]}
                    batch = MCQBatch(**payload)  # validate

                    # structural check (keep this)
                    for it in batch.items:
                        if not (0 <= it.answer_index < 4 and len(it.options) == 4):
                            raise HTTPException(500, "Invalid MCQ structure from model.")

                    result = batch.model_dump()
                    json_ok = 1
                    n_items =  len(result.get("items", []))

                else:

                    user_template = mlflow.genai.load_prompt("prompts:/quiz-generator-essay/2")
                    user = user_template.format(context=context, n=count, difficulty=difficulty)

                    payload = llm.chat_json(ESSAY_SYSTEM, user, temperature=0.4, max_tokens=1400)

                    # --- NORMALIZE ESSAY DIFFICULTY ---

                    items = payload.get("items", [])

                    norm_items = []

                    for it in items:
                        it = dict(it)

                        it["difficulty"] = _norm_difficulty(it.get("difficulty"))

                        norm_items.append(it)

                    payload["items"] = norm_items

                    # --- END NORMALIZE ---

                    batch = EssayBatch(**payload)

                    result = batch.model_dump()
                    json_ok = 1
                    n_items = len(result.get("items", []))

                # 8) Save to disk
                os.makedirs("outputs", exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = f"outputs/{qtype}_{ts}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                # log artifacts
                log_artifact(out_path, artifact_subdir="artifacts")
                # (optional) log normalized output separately
                log_json(result, "artifacts/result.json")

                return {
                    "status": "success",
                    "qtype": qtype,
                    "difficulty": difficulty,
                    "count": count,
                    "model": {"backend": config.LLM_BACKEND, "name": config.LLM_MODEL},
                    "saved_to": out_path,
                    "data": result,
                }

        finally:
            # Clean temp file
            try:
                os.unlink(path)
            except Exception:
                pass

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "backend": config.LLM_BACKEND,
            "model": config.LLM_MODEL
        }
    )
def main():
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)

if __name__ == "__main__":
    main()
