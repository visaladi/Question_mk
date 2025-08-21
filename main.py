# main.py
import os, json, tempfile, datetime
from typing import Optional, Literal

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
from core.prompts import MCQ_SYSTEM, MCQ_USER, ESSAY_SYSTEM, ESSAY_USER
from pathlib import Path
# Optional extras (safe import)
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

@app.get("/health")
def health():
    return {
        "ok": True,
        "backend": config.LLM_BACKEND,
        "model": config.LLM_MODEL,
        "rerank_enabled": config.USE_RERANK and Reranker is not None,
    }

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
        # 2) Extract + chunk
        pages = extract_pages(path)
        chunks = list(chunk_pages(pages, max_chars=1800, overlap=200))

        # 3) Build in-memory embed store
        store = EmbedStore()
        store.build(chunks)

        # 4) Retrieve and (optionally) rerank
        ctx_snips = store.topk(topic, k=6)
        if config.USE_RERANK and Reranker is not None:
            reranker = Reranker()
            ctx_snips = reranker.rerank(topic, ctx_snips, top_k=4)
        else:
            ctx_snips = ctx_snips[:4]

        # 5) Build context with page refs
        context = "\n\n".join([f"[pages {','.join(map(str,p))}]\n{t}" for t, p in ctx_snips])

        # 6) Static model from config.py
        llm = LLMClient(backend=config.LLM_BACKEND, model=config.LLM_MODEL)

        # 7) Generate
        if qtype == "mcq":
            user = MCQ_USER.format(context=context, n=count, difficulty=difficulty)
            payload = llm.chat_json(MCQ_SYSTEM, user, temperature=0.2, max_tokens=1200)
            batch = MCQBatch(**payload)  # validate
            # structural check
            for it in batch.items:
                if not (0 <= it.answer_index < 4 and len(it.options) == 4):
                    raise HTTPException(500, "Invalid MCQ structure from model.")
            result = batch.model_dump()
        else:
            user = ESSAY_USER.format(context=context, n=count)
            payload = llm.chat_json(ESSAY_SYSTEM, user, temperature=0.4, max_tokens=1200)

            # normalize difficulty values returned by the model
            for it in payload.get("items", []):
                d = it.get("difficulty")
                if isinstance(d, str):
                    it["difficulty"] = d.strip().lower()

            batch = EssayBatch(**payload)
            result = batch.model_dump()

        # 8) Save to disk
        os.makedirs("outputs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"outputs/{qtype}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

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
