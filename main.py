import os, json, tempfile
from typing import List, Tuple, Literal, Optional


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from flask.cli import load_dotenv
from pydantic import BaseModel



from core.pdf_utils import extract_pages, chunk_pages
from core.embed_store import EmbedStore
from core.schemas import MCQBatch, EssayBatch, GenerateReq
from core.llm_client import LLMClient
from core.prompts import MCQ_SYSTEM, MCQ_USER, ESSAY_SYSTEM, ESSAY_USER


# Optional extras (safe import)
try:
    from extras.ner_keywords import extract_keywords
except Exception:
    extract_keywords = None


try:
    from extras.rerank import Reranker
except Exception:
    Reranker = None


load_dotenv()


app = FastAPI(title="PDF → Questions (RAG)")
app.add_middleware(
CORSMiddleware,
allow_origins=["*"], allow_credentials=True,
allow_methods=["*"], allow_headers=["*"],
)


STORE: Optional[EmbedStore] = None
RERANKER: Optional["Reranker"] = None
LAST_DOC_INFO = {"chunks": 0}


class NERReq(BaseModel):
    top_k: int = 30
    min_len: int = 3


@app.get("/health")
def health():
    return {"ok": True, "chunks": LAST_DOC_INFO.get("chunks", 0)}


@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...), use_rerank: bool = False):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a .pdf file.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(await pdf.read())
        path = tf.name


    pages = extract_pages(path)
    chunks = list(chunk_pages(pages, max_chars=1800, overlap=200))
    os.unlink(path)


    global STORE, LAST_DOC_INFO, RERANKER
    STORE = EmbedStore()
    STORE.build(chunks)
    LAST_DOC_INFO = {"chunks": len(chunks)}


    if use_rerank and Reranker is not None:
        RERANKER = Reranker()


    return {"status": "indexed", "chunks": len(chunks)}

@app.post("/generate")
async def generate(req: GenerateReq):
    if STORE is None:
        raise HTTPException(400, "No document indexed. Upload a PDF first.")


    # Retrieve relevant chunks (optionally rerank with a cross-encoder)
    ctx_snips = STORE.topk(req.topic, k=6)
    if RERANKER is not None:
        ctx_snips = RERANKER.rerank(req.topic, ctx_snips, top_k=4)
    else:
        ctx_snips = ctx_snips[:4]


    context = "\n\n".join([f"[pages {','.join(map(str,p))}]\n{t}" for t, p in ctx_snips])


    llm = LLMClient(backend=req.llm_backend, model=req.llm_model)


    if req.qtype == "mcq":
        user = MCQ_USER.format(context=context, n=req.count, difficulty=req.difficulty)
        payload = llm.chat_json(MCQ_SYSTEM, user, temperature=0.2, max_tokens=1200)
        batch = MCQBatch(**payload) # validate
        for it in batch.items:
            if not (0 <= it.answer_index < 4 and len(it.options) == 4):
                raise HTTPException(500, "Invalid MCQ structure from model.")
        return batch.model_dump()
    else:
        user = ESSAY_USER.format(context=context, n=req.count)
        payload = llm.chat_json(ESSAY_SYSTEM, user, temperature=0.4, max_tokens=1200)
        batch = EssayBatch(**payload)
    return batch.model_dump()


@app.post("/keywords")
async def keywords(req: NERReq):
    if extract_keywords is None or STORE is None:
        raise HTTPException(400, "NER not available or no document indexed.")
        # build a synthetic doc to run NER over: join top chunks for 'full document'
    ctx_snips = STORE.topk("full document", k=12)
    text = "\n\n".join([t for t, _ in ctx_snips])
    kws = extract_keywords(text, top_k=req.top_k, min_len=req.min_len)
    return {"keywords": kws}
